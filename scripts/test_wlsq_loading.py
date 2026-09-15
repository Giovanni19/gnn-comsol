from gnn_comsol.data.loading import load_data


dataset_path = (
    "C:/Users/giovanni/.comsol/v64/llmatlab/"
    "channel2d_manual_stabilization_200R.mat"
)

data = load_data(
    dataset_path,
    skip_initial=0,
)

print("\n========================================")
print("WLSQ DATA LOADING TEST")
print("========================================")

print(f"Number of nodes : {data.num_nodes}")
print(f"Neighbors       : {len(data.neighbors)}")
print(f"G_wlsq          : {len(data.G_wlsq)}")
print(f"cell_index      : {data.cell_index.shape}")


# MATLAB node 1274 = Python node 1273
node = 1273

print("\n========================================")
print(f"EXAMPLE NODE {node}")
print("========================================")

print(f"Number of neighbors: {len(data.neighbors[node])}")

print("\nNeighbors:")
print(data.neighbors[node])

print("\nG_wlsq shape:")
print(data.G_wlsq[node].shape)

print("\nG_wlsq:")
print(data.G_wlsq[node])


print("\n========================================")
print("CELL CONNECTIVITY")
print("========================================")

print("First 5 cells:")
print(data.cell_index[:5])