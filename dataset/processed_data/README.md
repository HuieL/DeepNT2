## Data format

We build up graphs with `torch_geometric.data.Data`.

Run codes to process data to pyg Data:
```
# Transportation network:
python -m src.dataset.transportation

# Social network:
python -m src.dataset.social --input facebook 
python -m src.dataset.social --input twitter

# Computer network:
pending

# Synthetic network:
python -m src.dataset.synthetic
```
