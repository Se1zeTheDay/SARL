## Code implementation for paper: ```Advancing Rule Learning in Knowledge Graphs with Structure-Aware Graph Transformer``` 

## Preprocess

Run the following command to generate subgraph on your dataset:
```shell
python dataset.py -data=DATA -maxN=MAXN -padding=PADDING -jump=JUMP
```

## Train the model
```shell
python main.py -data=DATASET/DATA -jump=JUMP -padding=PADDING -batch_size=BATCH_SIZE -desc=DESC
```

## Decode the rules
```bash
python main.py -data=DATASET/DATA -jump=JUMP -padding=PADDING -batch_size=BATCH_SIZE -desc=DESC -ckpt=CKPT -decode_rule
```