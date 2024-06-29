import torch, numpy as np, pickle, random, time, argparse
from collections import defaultdict

import algo


def pad_1d_unsqueeze(x, padlen):
    # x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_sub_unsqueeze(x, padlen):
    # x = x + 1  # pad id = 0
    x = x.unsqueeze(-1)
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_deg_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_path_unsqueeze(x, padlen1, padlen2, padlen3, rels_len):
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = torch.full((padlen1, padlen2, padlen3, xlen4), rels_len, dtype=x.dtype)
        # new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


class DataBase():
    def __init__(self, data_path, rel_pad=0, subgraph='') -> None:
        with open(f'{data_path}/entities.txt') as e, open(f'{data_path}/relations.txt') as r:
            self.ents = ['<pad>'] + [x.strip() for x in e.readlines()]
            self.rels = [x.strip() for x in r.readlines()]
            self.pos_rels = len(self.rels)
            self.rels += ['inv_' + x for x in self.rels] + ['<slf>']
            self.e2id = {self.ents[i]: i for i in range(len(self.ents))}
            self.r2id = {self.rels[i]: i for i in range(len(self.rels))}
            self.id2r = {i: self.rels[i] for i in range(len(self.rels))}
            self.id2e = {i: self.ents[i] for i in range(len(self.ents))}
        self.data = {}
        with open(f'{data_path}/train.txt') as f:
            train = [item.strip().split('\t') for item in f.readlines()]
            self.data['train'] = list({(self.e2id[h], self.r2id[r], self.e2id[t]) for h, r, t in train})
        with open(f'{data_path}/test.txt') as f:
            test = [item.strip().split('\t') for item in f.readlines()]
            self.data['test'] = list({(self.e2id[h], self.r2id[r], self.e2id[t]) for h, r, t in test})
        with open(f'{data_path}/valid.txt') as f:
            valid = [item.strip().split('\t') for item in f.readlines()]
            self.data['valid'] = list({(self.e2id[h], self.r2id[r], self.e2id[t]) for h, r, t in valid})

        indices = [[] for _ in range(self.pos_rels)]
        values = [[] for _ in range(self.pos_rels)]
        for h, r, t in self.data['train']:
            indices[r].append((h, t))
            values[r].append(1)
        indices = [torch.LongTensor(x).T for x in
                   indices]
        values = [torch.FloatTensor(x) for x in values]
        size = torch.Size([len(self.ents), len(self.ents)])
        self.relations = [torch.sparse.FloatTensor(indices[i], values[i], size).coalesce() for i in
                          range(self.pos_rels)]

        self.filtered_dict = defaultdict(set)
        triplets = self.data['train'] + self.data['valid'] + self.data['test']
        for triplet in triplets:
            self.filtered_dict[(triplet[0], triplet[1])].add(triplet[2])
            self.filtered_dict[(triplet[2], triplet[1] + self.pos_rels)].add(triplet[0])

        self.neighbors = defaultdict(dict)
        for h, r, t in self.data['train']:
            try:
                self.neighbors[h][r].add(t)
            except KeyError:
                self.neighbors[h][r] = set([t])
            try:
                self.neighbors[t][r + self.pos_rels].add(h)
            except KeyError:
                self.neighbors[t][r + self.pos_rels] = set([h])
        for h in self.neighbors:
            self.neighbors[h] = {r: list(ts) for r, ts in self.neighbors[h].items()}

        self.nebor_relation = torch.ones(len(self.e2id), len(self.r2id))
        for h, r, t in self.data['train']:
            self.nebor_relation[h][r] += 1
            self.nebor_relation[t][r + self.pos_rels] += 1
        for e in self.e2id.values():
            if e not in self.neighbors.keys():
                self.nebor_relation[e][2 * self.pos_rels] += 1
        self.nebor_relation = torch.log(self.nebor_relation)
        self.nebor_relation /= self.nebor_relation.sum(1).unsqueeze(1)

        if subgraph:
            with open(subgraph, 'rb') as db:
                self.subgraph = pickle.load(db)

    def getinfo(self):
        return len(self.ents), len(self.rels)

    def extract_without_token(self, head, JUMP, MAXN, PADDING):
        subgraph = [head]
        relation = []
        length = [0]
        for jump in range(JUMP):
            length.append(len(subgraph))
            for parent in range(length[jump], length[jump + 1]):
                for r in self.neighbors[subgraph[parent]]:
                    if len(self.neighbors[subgraph[parent]][r]) > MAXN:
                        print(f'J{subgraph[parent]}-{r}-{len(self.neighbors[subgraph[parent]][r])}', end=' ')
                        # continue
                        random.shuffle(self.neighbors[subgraph[parent]][r])
                    for t in self.neighbors[subgraph[parent]][r][:MAXN]:
                        try:
                            pos = subgraph.index(t)
                            relation.append((parent, pos, r))
                        except ValueError:
                            subgraph.append(t)
                            pos = subgraph.index(t)
                            relation.append((parent, pos, r))
        length.append(len(subgraph))
        if length[-1] > PADDING or not relation:  # subgraph is too big
            subgraph = subgraph[:PADDING]
            length[-1] = len(subgraph)
        RELA = set()
        rel_dict = defaultdict(dict)
        for i, j, r in relation:
            if i < PADDING and j < PADDING:
                rel_dict[i][j] = r
                RELA.add((i, j, r))
                inv_r = r + self.pos_rels * ((r < self.pos_rels) * 2 - 1)
                rel_dict[j][i] = inv_r
                RELA.add((j, i, inv_r))

        # get indegree / outdegree / shortest path
        adjacent = np.zeros((len(subgraph), len(subgraph)))
        indeg = np.zeros(len(subgraph))
        outdeg = np.zeros(len(subgraph))
        for i, j, r in RELA:
            # i -> j with r
            adjacent[i, j] = 1
            outdeg[i] += 1
            indeg[j] += 1

        dist, all_path_result = algo.floyd(adjacent)
        if (np.max(dist) == 0x3f3f3f3f):
            m = 10
        else:
            m = np.max(dist)
        shortest_path = np.full((len(subgraph), len(subgraph), m, 1), len(self.rels))
        for i in range(len(subgraph)):
            for j in range(len(subgraph)):
                for k in range(len(all_path_result[i][j])):
                    # all_path_result[i][j][k - 1] -> all_path_result[i][j][k]
                    if k == 0:
                        shortest_path[i][j][k] = rel_dict[i][all_path_result[i][j][k]]
                    elif k == len(all_path_result[i][j]) - 1:
                        shortest_path[i][j][k] = rel_dict[all_path_result[i][j][k]][j]
                    else:
                        shortest_path[i][j][k] = rel_dict[all_path_result[i][j][k - 1]][all_path_result[i][j][k]]

        attn_bias = np.zeros((len(subgraph) + 1, len(subgraph) + 1), dtype=float)

        return subgraph, np.array(list(RELA)).T, length, indeg, outdeg, dist, shortest_path, attn_bias


class pickleDataset(torch.utils.data.Dataset):
    def __init__(self, database, opt, mode='train'):
        super().__init__()
        self.triples = database.data[mode]
        self.neighbors = database.neighbors
        self.nrels = len(database.rels)
        self.pos_rels = self.nrels // 2
        self.padding = opt.padding
        self.subgraph = database.subgraph

    def __getitem__(self, index):
        H, R, T = self.triples[index]
        sub1, rela1, trg1, tails1, leng1, indeg1, outdeg1, dist1, shortest_path1, attn_bias1 = self.getsubgraph(H, R, T)
        sub2, rela2, trg2, tails2, leng2, indeg2, outdeg2, dist2, shortest_path2, attn_bias2 = self.getsubgraph(T,
                                                                                                                R + self.pos_rels,
                                                                                                                H)
        return sub1, rela1, trg1, tails1, leng1, indeg1, outdeg1, dist1, shortest_path1, attn_bias1, sub2, rela2, trg2, tails2, leng2, indeg2, outdeg2, dist2, shortest_path2, attn_bias2

    def getsubgraph(self, H, R, T):
        subgraph, relations, length, indeg, outdeg, dist, shortest_path, attn_bias = self.subgraph[H]
        assert (subgraph.index(H) == 0)
        rela_mat = torch.zeros(self.padding, self.padding, self.nrels)
        rela_mat[relations] = 1

        try:
            t = subgraph.index(T)
            inv_r = R + self.pos_rels * ((R < self.pos_rels) * 2 - 1)
            rela_mat[subgraph.index(H), t, R] = 0
            rela_mat[t, subgraph.index(H), inv_r] = 0
        except ValueError:
            pass

        return (torch.LongTensor(subgraph), torch.FloatTensor(rela_mat), torch.tensor([H, R, T]), torch.LongTensor([0]),
                torch.LongTensor(length)
                , torch.LongTensor(indeg), torch.LongTensor(outdeg), torch.LongTensor(dist),
                torch.LongTensor(shortest_path), torch.FloatTensor(attn_bias))

    def __len__(self) -> int:
        return len(self.triples)

    @staticmethod
    def collate_fn(data):
        # todo make max_node_num configurable
        max_node_num = 40
        rels_len = 93
        subs_1 = torch.cat([pad_sub_unsqueeze(i, max_node_num) for i in [d[0] for d in data]])
        subs_2 = torch.cat([pad_sub_unsqueeze(i, max_node_num) for i in [d[10] for d in data]])
        subs = torch.cat([subs_1, subs_2], dim=0)
        relas = torch.stack([d[1] for d in data] + [d[11] for d in data], dim=0)
        trgs = torch.stack([d[2] for d in data] + [d[12] for d in data], dim=0)
        tails = torch.stack([d[3] for d in data] + [d[13] for d in data], dim=0)
        lengs = torch.stack([d[4] for d in data] + [d[14] for d in data], dim=0)

        indeg_1 = torch.cat([pad_deg_unsqueeze(i, max_node_num) for i in [d[5] for d in data]])
        indeg_2 = torch.cat([pad_deg_unsqueeze(i, max_node_num) for i in [d[15] for d in data]])
        indeg = torch.cat([indeg_1, indeg_2], dim=0)

        outdeg_1 = torch.cat([pad_deg_unsqueeze(i, max_node_num) for i in [d[6] for d in data]])
        outdeg_2 = torch.cat([pad_deg_unsqueeze(i, max_node_num) for i in [d[16] for d in data]])
        outdeg = torch.cat([outdeg_1, outdeg_2], dim=0)

        dist_1 = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in [d[7] for d in data]])
        dist_2 = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in [d[17] for d in data]])
        dist = torch.cat([dist_1, dist_2], dim=0)

        shortest_path_unpad_1 = [d[8] for d in data]
        shortest_path_unpad_2 = [d[18] for d in data]
        max_dist = max(
            [tens.size(2) for tens in shortest_path_unpad_1] + [tens.size(2) for tens in shortest_path_unpad_2])
        shortest_path_1 = torch.cat(
            [pad_path_unsqueeze(i, max_node_num, max_node_num, max_dist, rels_len) for i in shortest_path_unpad_1]
        )
        shortest_path_2 = torch.cat(
            [pad_path_unsqueeze(i, max_node_num, max_node_num, max_dist, rels_len) for i in shortest_path_unpad_1]
        )
        shortest_path = torch.cat([shortest_path_1, shortest_path_2], dim=0)

        attn_bias_1 = torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in [d[9] for d in data]])
        attn_bias_2 = torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in [d[19] for d in data]])
        attn_bias = torch.cat([attn_bias_1, attn_bias_2], dim=0)
        return subs, relas, trgs, tails, lengs, indeg, outdeg, dist, shortest_path, attn_bias


def main(jump, base_data, path, maxn, padding):
    subgraph = dict()
    cnt = []
    false = []
    for head in range(1, len(base_data.ents)):  # 0 for <pad>
        res = base_data.extract_without_token(head, jump, maxn, padding)  # head, JUMP, MAXN, PADDING
        try:
            print(f'{head}\tlen:{res[2]}')
            subgraph[head] = res
            cnt.append(res[2][-1])
        except Exception:
            false.append(base_data.ents[head])
    print(len(false), false)
    with open(f'{path}/subgraph{jump}_{padding}_{maxn}', 'wb') as db:
        pickle.dump(subgraph, db)
        print(sum(cnt) / len(cnt))
    lis = {i: 0 for i in range(padding + 1)}
    for item in cnt:
        lis[item] += 1
    print({k: v for k, v in lis.items() if v})


if __name__ == '__main__':
    """
    Make sure to change the 'max_node_num' and 'rels_len' in collate_fn when you have generated different subgraphs.
    For example: for fb15k-237, we extract the subgraph with padding = 100 and fb15k-237 has 435 relations in total, 
    so we should change the 'max_node_num' and 'rels_len' to 100 and 435 respectively.
    
    We haven't made these parameters configurable at this time and we will finish it in the future.
    """
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-data', type=str, required=True)
    parser.add_argument('-maxN', type=int, required=True)
    parser.add_argument('-jump', type=int, required=True)
    parser.add_argument('-padding', type=int, required=True)
    opt = parser.parse_args()
    # wn18rr MAXN=10
    # umls   MAXN=40
    # 237    MAXN=70
    # 237    MaxN=40
    path = opt.data
    base_data = DataBase(path)
    # main(1, base_data, path, 100, 40)
    main(opt.jump, base_data, path, opt.maxN, opt.padding)
    # main(3, base_data, path, 10, 100)
    # main(3, base_data)
