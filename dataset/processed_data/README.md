## Data format

We build up graphs with `torch_geometric.data.Data`.

Run codes to process data to pyg Data:
```
# Transportation network:
python tntp_to_pyg_converter.py

# Social network:
python process_social_network.py --input facebook 
python process_social_network.py --input twitter

# Computer network:
pending

# Synthetic network:
pending
```
