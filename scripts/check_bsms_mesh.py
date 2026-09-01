import numpy as np

from gnn_comsol.graph.bsms import BistrideMultiLayerGraph

from gnn_comsol.data.loading import load_data


DATASET_PATH = (
    r"C:\Users\giovanni\.comsol\v64\llmatlab"
    r"\channel2d_gnn_dataset.mat"
)


# ---------------------------------------------------------
# 1. Load COMSOL dataset
# ---------------------------------------------------------

raw = load_data(DATASET_PATH)

edge_index = raw.edge_index
pos = raw.pos

num_nodes = raw.num_nodes


print("Original COMSOL graph")
print("---------------------")
print("Nodes:", num_nodes)
print("Edges:", edge_index.shape[1])
print("Position shape:", pos.shape)
print()


# ---------------------------------------------------------
# 2. Build BSMS hierarchy
# ---------------------------------------------------------

num_layers = 2

multi_layer_graph = BistrideMultiLayerGraph(
    edge_index,
    num_layers,
    num_nodes,
    pos,
)

m_gs, m_flat_es, m_ids = (
    multi_layer_graph.get_multi_layer_graphs()
)


# ---------------------------------------------------------
# 3. Inspect hierarchy
# ---------------------------------------------------------

print("BSMS hierarchy")
print("--------------")

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
# 4. Inspect pooling
# ---------------------------------------------------------

for level, ids in enumerate(m_ids):

    print(
        f"Pooling {level} -> {level + 1}: "
        f"{len(ids)} nodes retained"
    )
