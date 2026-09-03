from gnn_comsol import data as gdata


raw = gdata.load_data(
    r"C:\Users\giovanni\.comsol\v64\llmatlab\channel2d_physics_variables_gnn_dataset.mat",
    skip_initial=0,
)


splits = gdata.split_dataset(
    raw,
    mode="temporal",
    train_fraction=0.70,
    val_fraction=0.15,
    gap=1,
)


print("\nRAW")
print("X:", raw.X_input.shape)
print("physics:", raw.physics_features.shape)


print("\nTRAIN")
print("X:", splits.train.X.shape)
print("physics:", splits.train.physics_features.shape)


print("\nVAL")
print("X:", splits.val.X.shape)
print("physics:", splits.val.physics_features.shape)


print("\nTEST")
print("X:", splits.test.X.shape)
print("physics:", splits.test.physics_features.shape)