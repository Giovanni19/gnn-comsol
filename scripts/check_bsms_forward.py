import numpy as np
import torch

from gnn_comsol.data.loading import load_data
from gnn_comsol.graph.bsms import BistrideMultiLayerGraph
from gnn_comsol.models.bsms_pressure import PressureBSMSGNN


DATASET_PATH = (
    r"C:\Users\giovanni\.comsol\v64\llmatlab"
    r"\channel2d_gnn_dataset.mat"
)


# ---------------------------------------------------------
# 1. Load real COMSOL mesh
# ---------------------------------------------------------

raw = load_data(DATASET_PATH)

edge_index = raw.edge_index
pos = raw.pos
num_nodes = raw.num_nodes

print("Original graph")
print("Nodes:", num_nodes)
print("Edges:", edge_index.shape[1])
print("Position shape:", pos.shape)
print()


# ---------------------------------------------------------
# 2. Build BSMS hierarchy
# ---------------------------------------------------------

unet_depth = 2

multi_layer_graph = BistrideMultiLayerGraph(
    edge_index,
    unet_depth,
    num_nodes,
    pos,
)

_, m_flat_es, m_ids = (
    multi_layer_graph.get_multi_layer_graphs()
)

print("BSMS hierarchy")

current_num_nodes = num_nodes

for level, edges in enumerate(m_flat_es):

    print(
        f"Level {level}: "
        f"{current_num_nodes} nodes, "
        f"{edges.shape[1]} edges"
    )

    if level < len(m_ids):
        current_num_nodes = len(m_ids[level])

print()


# ---------------------------------------------------------
# 3. Convert BSMS structures to PyTorch
# ---------------------------------------------------------

edge_indices = [
    torch.tensor(
        edges,
        dtype=torch.long,
    )
    for edges in m_flat_es
]

pool_indices = [
    torch.tensor(
        indices,
        dtype=torch.long,
    )
    for indices in m_ids
]

pos_tensor = torch.tensor(
    pos,
    dtype=torch.float32,
)


# ---------------------------------------------------------
# 4. Create synthetic node features
# ---------------------------------------------------------

# For now the exact number is not important.
# We only want to test the architecture.
num_in = 4

x = torch.randn(
    num_nodes,
    num_in,
)


# ---------------------------------------------------------
# 5. Create pressure network
# ---------------------------------------------------------

model = PressureBSMSGNN(
    num_in=num_in,
    latent_dim=64,
    unet_depth=unet_depth,
    hidden_layers=2,
    edge_indices=edge_indices,
    pool_indices=pool_indices,
    pos=pos_tensor,
    pos_dim=2,
)


# ---------------------------------------------------------
# 6. Forward pass
# ---------------------------------------------------------

model.eval()

with torch.no_grad():

    prediction = model(x)


# ---------------------------------------------------------
# 7. Check result
# ---------------------------------------------------------

print("Input shape: ", x.shape)
print("Output shape:", prediction.shape)

expected_shape = (
    num_nodes,
    1,
)

if prediction.shape != expected_shape:

    raise RuntimeError(
        f"Expected output shape {expected_shape}, "
        f"got {tuple(prediction.shape)}."
    )

if not torch.isfinite(prediction).all():

    raise RuntimeError(
        "Prediction contains NaN or Inf."
    )

print()
print("BSMS forward pass successful.")


# ---------------------------------------------------------
# 8. Test batch
# ---------------------------------------------------------

batch_size = 8

x_batch = torch.randn(
    batch_size,
    num_nodes,
    num_in,
)

with torch.no_grad():
    prediction_batch = model(x_batch)

print()
print("Batch input shape: ", x_batch.shape)
print("Batch output shape:", prediction_batch.shape)

expected_batch_shape = (
    batch_size,
    num_nodes,
    1,
)

if prediction_batch.shape != expected_batch_shape:
    raise RuntimeError(
        f"Expected batch output shape {expected_batch_shape}, "
        f"got {tuple(prediction_batch.shape)}."
    )

if not torch.isfinite(prediction_batch).all():
    raise RuntimeError(
        "Batch prediction contains NaN or Inf."
    )

print()
print("BSMS batch forward pass successful.")
