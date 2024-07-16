## Data format

We build up graphs with `torch_geometric.data.Data`.

Run codes to process data to pyg Data:
```
# Transportation network:
python src/dataset/transportation.py

# Social network:
python src/dataset/social.py --input facebook 
python src/dataset/social.py --input twitter

# Computer network:
pending

# Synthetic network:
pending
```
