import os
import copy
import random
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

import dataset_preparation as data
import Visualizations as plot
from training import train_network
import inference as inference

# ============================================================
# Reproducibility
# ============================================================

SEED = 68

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# Paths
# ============================================================

# Dataset
dataset_path = (
    r"C:\Users\giovanni\.comsol\v64\llmatlab"
    r"\channel2d_gnn_dataset.mat"
)

# Directory where all grid-search models are saved
save_dir = r"networks\grid_search"

os.makedirs(
    save_dir,
    exist_ok=True
)

# Best model
save_best_model = r"networks\GCN_best_model.pth"

os.makedirs(
    os.path.dirname(save_best_model),
    exist_ok=True
)


# ============================================================
# Load data
# ============================================================

X_input, Y_target, edge_index, edge_weight, delta_t, h = (
    data.load_data(dataset_path)
)



# ============================================================
# Train / validation / test split
# ============================================================

(
    X_train,
    Y_train,
    X_val,
    Y_val,
    X_test,
    Y_test,
    dt_train,
    dt_val,
    dt_test
) = data.split_dataset(
    X_input,
    Y_target,
    delta_t,
    n_train=70,
    n_val=15
)


# ============================================================
# Normalization parameters
# ============================================================

x_mean, x_std = (
    data.compute_normalization_parameters(
        X_train
    )
)


# ============================================================
# Normalize datasets
# ============================================================

(
    X_train_norm,
    X_val_norm,
    X_test_norm
) = data.normalize_datasets(
    X_train,
    Y_train,
    X_val,
    Y_val,
    X_test,
    Y_test,
    x_mean,
    x_std
)
Y_train_norm = Y_train

Y_test_norm = Y_test

Y_val_norm = Y_val

##Proviamo a inserire la derivata temporale per vedere se le cose migliorano. Proviamo inoltre a 
# dare il timestep in pasto al mio neural network




X_train_norm = data.add_time_feature(
    X_train_norm,
    dt_train
)

X_val_norm = data.add_time_feature(
    X_val_norm,
    dt_val
)

X_test_norm = data.add_time_feature(
    X_test_norm,
    dt_test
)


# ============================================================
# Convert to PyTorch tensors
# ============================================================

(
    X_train,
    Y_train,
    X_val,
    Y_val,
    X_test,
    Y_test,
    edge_index,
    edge_weight
) = data.convert_to_torch(
    X_train_norm,
    Y_train_norm,
    X_val_norm,
    Y_val_norm,
    X_test_norm,
    Y_test_norm,
    edge_index,
    edge_weight
)


# ============================================================
# Create graph datasets
# ============================================================

train_dataset = data.create_graph_dataset(
    X_train,
    Y_train,
    edge_index,
    edge_weight
)

val_dataset = data.create_graph_dataset(
    X_val,
    Y_val,
    edge_index,
    edge_weight
)

test_dataset = data.create_graph_dataset(
    X_test,
    Y_test,
    edge_index,
    edge_weight
)


# ============================================================
# GCN architecture
# ============================================================

class GCNNet(nn.Module):

    def __init__(
        self,
        num_in,
        num_out,
        num_neurons,
        num_layers,
        dropout
    ):

        super().__init__()

        # List of GCN layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            GCNConv(
                num_in,
                num_neurons
            )
        )

        # Hidden layers
        for _ in range(num_layers - 1):

            self.layers.append(
                GCNConv(
                    num_neurons,
                    num_neurons
                )
            )

        # Activation
        self.activation = nn.LeakyReLU()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.out_layer = nn.Linear(
            num_neurons,
            num_out
        )


    def forward(self, graph):

        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight

        for layer in self.layers:

            x = layer(
                x,
                edge_index,
                edge_weight=edge_weight
            )

            x = self.activation(x)

            x = self.dropout(x)

        x = self.out_layer(x)

        return x


# ============================================================
# Grid search settings
# ============================================================

neurons_list = [
    64,
]

layers_list = [
    3,
]

lr_list = [
    1e-3,
]

batch_size_list = [
    8,
]

dropout_list = [
    0
]


# Number of epochs for each configuration
num_epochs = 200


# Loss function
criterion = nn.MSELoss()


# ============================================================
# Initialise best values
# ============================================================

best_val_rmse = float("inf")

best_config = None

best_model_state = None

results_list = []


# ============================================================
# Grid search
# ============================================================

for (
    num_neurons,
    num_layers,
    lr,
    batch_size,
    dropout
) in itertools.product(
    neurons_list,
    layers_list,
    lr_list,
    batch_size_list,
    dropout_list
):

    print("\n============================================")

    print(
        f"Testing: "
        f"neurons={num_neurons}, "
        f"layers={num_layers}, "
        f"lr={lr}, "
        f"batch={batch_size}, "
        f"dropout={dropout}"
    )

    print("============================================")


    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # ========================================================
    # Model
    # ========================================================

    net = GCNNet(
        num_in=4,
        num_out=3,
        num_neurons=num_neurons,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = optim.Adam(
        net.parameters(),
        lr=lr,
        weight_decay=1e-5
    )


    # ========================================================
    # Training
    # ========================================================

    (
        model_state,
        train_rmse_history,
        val_rmse_history
    ) = train_network(
        net,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs,
        device
    )


    # ========================================================
    # Best validation RMSE for this configuration
    # ========================================================

    final_val_rmse = min(
        val_rmse_history
    )

    print(
        f"Best Validation RMSE: "
        f"{final_val_rmse:.6e}"
    )


    # ========================================================
    # Load best model of this configuration
    # ========================================================

    net.load_state_dict(
        model_state
    )

    net.eval()


    # ========================================================
    # Test evaluation
    # ========================================================

    test_losses = []

    with torch.no_grad():

        for batch in test_loader:

            batch = batch.to(device)

            preds = net(batch)

            loss = torch.sqrt(
                criterion(
                    preds,
                    batch.y
                )
            )

            test_losses.append(
                loss.item()
            )


    test_rmse = np.mean(
        test_losses
    )

    print(
        f"Test RMSE: "
        f"{test_rmse:.6e}"
    )


    # ========================================================
    # Save model for this configuration
    # ========================================================

    model_name = (
        f"GCN_"
        f"n{num_neurons}_"
        f"l{num_layers}_"
        f"lr{lr}_"
        f"b{batch_size}_"
        f"d{dropout}.pth"
    )

    model_path = os.path.join(
        save_dir,
        model_name
    )

    torch.save(
        model_state,
        model_path
    )


    # ========================================================
    # Store results
    # ========================================================

    results_list.append(
        {
            "neurons": num_neurons,
            "layers": num_layers,
            "learning_rate": lr,
            "batch_size": batch_size,
            "dropout": dropout,
            "val_rmse": final_val_rmse,
            "test_rmse": test_rmse,
            "model_path": model_path
        }
    )


    # ========================================================
    # Track globally best model
    # ========================================================

    if final_val_rmse < best_val_rmse:

        best_val_rmse = final_val_rmse

        best_config = (
            num_neurons,
            num_layers,
            lr,
            batch_size,
            dropout
        )

        best_model_state = copy.deepcopy(
            model_state
        )


# ============================================================
# Best model summary
# ============================================================

print("\n============================================")
print("BEST CONFIGURATION")
print("============================================")

print(
    f"Neurons: {best_config[0]}"
)

print(
    f"Layers: {best_config[1]}"
)

print(
    f"Learning rate: {best_config[2]}"
)

print(
    f"Batch size: {best_config[3]}"
)

print(
    f"Dropout: {best_config[4]}"
)

print(
    f"Best Validation RMSE: "
    f"{best_val_rmse:.6e}"
)


# ============================================================
# Create best model
# ============================================================

best_model = GCNNet(
    num_in=4,
    num_out=3,
    num_neurons=best_config[0],
    num_layers=best_config[1],
    dropout=best_config[4]
).to(device)

best_model.load_state_dict(
    best_model_state
)


# ============================================================
# Save best model
# ============================================================

torch.save(
    best_model.state_dict(),
    save_best_model
)

print(
    f"\nBest model saved at: "
    f"{save_best_model}"
)


# ============================================================
# Save grid-search results to Excel
# ============================================================

results_df = pd.DataFrame(
    results_list
)

results_df = results_df.sort_values(
    by="val_rmse"
)

excel_path = os.path.join(
    save_dir,
    "grid_search_results.xlsx"
)

results_df.to_excel(
    excel_path,
    index=False
)

print(
    f"Results saved to: "
    f"{excel_path}"
)


# ============================================================
# Plot last training run
# ============================================================

plot.plot_loss(
    train_rmse_history,
    val_rmse_history
)



# ============================================================
# Select timestep for inference
# ============================================================

timestep = 90

X_current = X_input[timestep]
Y_true = Y_target[timestep]
delta_t_current = delta_t[timestep]


# ============================================================
# Predict next timestep
# ============================================================

Y_pred = inference.predict_next_timestep(
    model=best_model,
    X=X_current,
    delta_t=delta_t_current,
    edge_index=edge_index,
    edge_weight=edge_weight,
    x_mean=x_mean,
    x_std=x_std,
    device=device
)


# ============================================================
# Compute MSE
# ============================================================

mse_total, mse_u, mse_v, mse_p = inference.compute_mse(
    Y_true,
    Y_pred
)

# ============================================================
# Compute RELATIVE ERROR
# ============================================================


relative_error_u, relative_error_v, relative_error_p = (
    inference.compute_relative_error(
        Y_true,
        Y_pred
    )
)

# ============================================================
# Print results
# ============================================================

print("\n============================================")
print("INFERENCE RESULTS")
print("============================================")

print(f"Input timestep: {timestep}")
print(f"Target timestep: {timestep + 1}")

print(f"\nTotal MSE: {mse_total:.6e}")
print(f"MSE u:     {mse_u:.6e}")
print(f"MSE v:     {mse_v:.6e}")
print(f"MSE p:     {mse_p:.6e}")


print(f"relative error u:     {relative_error_u:.6e}")
print(f"relative error v:     {relative_error_v:.6e}")
print(f"relative error p:     {relative_error_p:.6e}")





import matplotlib.pyplot as plt
import numpy as np


def plot_pressure_statistics(X):

    p = X[:, :, 2]

    p_mean = np.mean(p, axis=1)
    p_std = np.std(p, axis=1)
    p_min = np.min(p, axis=1)
    p_max = np.max(p, axis=1)

    timesteps = np.arange(X.shape[0])

    plt.figure(figsize=(9, 5))

    plt.plot(timesteps, p_mean, label="Mean pressure")
    plt.plot(timesteps, p_max, label="Max pressure")
    plt.plot(timesteps, p_min, label="Min pressure")

    plt.xlabel("Timestep")
    plt.ylabel("Pressure")
    plt.title("Pressure statistics over time")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_pressure_statistics(X_input)


###CERCHIAMO DI CAPIRE COSA DIAVOLO SUCCEDE CON LA TRAIN-VAL LOSS

def evaluate_dataset(net, loader, criterion, device):

    net.eval()

    total_losses = []
    u_losses = []
    v_losses = []
    p_losses = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            preds = net(batch)

            total_losses.append(
                criterion(preds, batch.y).item()
            )

            u_losses.append(
                criterion(
                    preds[:, 0],
                    batch.y[:, 0]
                ).item()
            )

            v_losses.append(
                criterion(
                    preds[:, 1],
                    batch.y[:, 1]
                ).item()
            )

            p_losses.append(
                criterion(
                    preds[:, 2],
                    batch.y[:, 2]
                ).item()
            )

    return (
        np.mean(total_losses),
        np.mean(u_losses),
        np.mean(v_losses),
        np.mean(p_losses)
    )


train_eval = evaluate_dataset(
    best_model,
    train_loader,
    criterion,
    device
)

val_eval = evaluate_dataset(
    best_model,
    val_loader,
    criterion,
    device
)

test_eval = evaluate_dataset(
    best_model,
    test_loader,
    criterion,
    device
)


print("\nFINAL EVALUATION")

print(
    "Train:",
    f"total={train_eval[0]:.6e}",
    f"u={train_eval[1]:.6e}",
    f"v={train_eval[2]:.6e}",
    f"p={train_eval[3]:.6e}"
)

print(
    "Val:  ",
    f"total={val_eval[0]:.6e}",
    f"u={val_eval[1]:.6e}",
    f"v={val_eval[2]:.6e}",
    f"p={val_eval[3]:.6e}"
)

print(
    "Test: ",
    f"total={test_eval[0]:.6e}",
    f"u={test_eval[1]:.6e}",
    f"v={test_eval[2]:.6e}",
    f"p={test_eval[3]:.6e}"
)

##%

"""
ATTENZIONE, ECCO IL RISULTATO!!! il problema e la pressione, probabilmente (anche a livello statistico), i casi piu difficili sono rappresentati nel training set

FINAL EVALUATION
Train: total=1.103403e-01 u=2.787864e-04 v=6.847041e-05 p=3.306736e-01
Val:   total=3.567309e-04 u=1.031607e-05 v=2.085820e-05 p=1.039018e-03
Test:  total=5.723553e+00 u=1.347894e-02 v=2.067363e-03 p=1.715511e+01

"""
