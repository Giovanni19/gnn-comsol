import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

import dataset_preparation as data
import Visualizations as plot
from training_separate import train_network
import inference_separate as inference


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

# Networks directory
save_dir = r"networks"

os.makedirs(
    save_dir,
    exist_ok=True
)

# Best velocity model
save_velocity_model = (
    r"networks\GCN_velocity_best_model.pth"
)

# Best pressure model
save_pressure_model = (
    r"networks\GCN_pressure_best_model.pth"
)


# ============================================================
# Load data
# ============================================================

(
    X_input,
    Y_target,
    edge_index,
    edge_weight,
    delta_t,
    h
) = data.load_data(
    dataset_path
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
    train_fraction=0.70,
    val_fraction=0.15
)


# ============================================================
# Normalization parameters
# ============================================================

x_mean, x_std, dt_mean, dt_std = (
    data.compute_normalization_parameters(
        X_train,
        dt_train
    )
)


# ============================================================
# Normalize input datasets
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

dt_train_norm = (dt_train - dt_mean) / dt_std
dt_val_norm   = (dt_val - dt_mean) / dt_std
dt_test_norm  = (dt_test - dt_mean) / dt_std

# ============================================================
# Targets remain in physical units
# ============================================================

Y_train_norm = Y_train
Y_val_norm = Y_val
Y_test_norm = Y_test


# ============================================================
# Time features for VELOCITY network
#
# Input:
# [u_norm, v_norm, p_norm, delta_t]
# ============================================================

X_train_velocity = data.add_time_feature(
    X_train_norm,
    dt_train_norm
)

X_val_velocity = data.add_time_feature(
    X_val_norm,
    dt_val_norm
)

X_test_velocity = data.add_time_feature(
    X_test_norm,
    dt_test_norm
)

# ============================================================
# Time + Fourier features for PRESSURE network
#
# Input:
# [
#   u_norm,
#   v_norm,
#   p_norm,
#   delta_t,
#   sin(2*pi*dt),
#   cos(2*pi*dt),
#   sin(4*pi*dt),
#   cos(4*pi*dt),
#   sin(8*pi*dt),
#   cos(8*pi*dt),
#   sin(16*pi*dt),
#   cos(16*pi*dt)
# ]
# ============================================================

X_train_pressure = data.add_time_fourier_features(
    X_train_norm,
    dt_train_norm,
    num_frequencies=4
)

X_val_pressure = data.add_time_fourier_features(
    X_val_norm,
    dt_val_norm,
    num_frequencies=4
)

X_test_pressure = data.add_time_fourier_features(
    X_test_norm,
    dt_test_norm,
    num_frequencies=4
)


# ============================================================
# Convert to PyTorch tensors
# ============================================================

X_train_velocity = torch.tensor(
    X_train_velocity,
    dtype=torch.float32
)

X_val_velocity = torch.tensor(
    X_val_velocity,
    dtype=torch.float32
)

X_test_velocity = torch.tensor(
    X_test_velocity,
    dtype=torch.float32
)


X_train_pressure = torch.tensor(
    X_train_pressure,
    dtype=torch.float32
)

X_val_pressure = torch.tensor(
    X_val_pressure,
    dtype=torch.float32
)

X_test_pressure = torch.tensor(
    X_test_pressure,
    dtype=torch.float32
)


Y_train = torch.tensor(
    Y_train_norm,
    dtype=torch.float32
)

Y_val = torch.tensor(
    Y_val_norm,
    dtype=torch.float32
)

Y_test = torch.tensor(
    Y_test_norm,
    dtype=torch.float32
)


edge_index = torch.tensor(
    edge_index,
    dtype=torch.long
)

edge_weight = torch.tensor(
    edge_weight,
    dtype=torch.float32
)


# ============================================================
# Create graph datasets - VELOCITY
# ============================================================

train_dataset_velocity = data.create_graph_dataset(
    X_train_velocity,
    Y_train,
    edge_index,
    edge_weight
)

val_dataset_velocity = data.create_graph_dataset(
    X_val_velocity,
    Y_val,
    edge_index,
    edge_weight
)

test_dataset_velocity = data.create_graph_dataset(
    X_test_velocity,
    Y_test,
    edge_index,
    edge_weight
)


# ============================================================
# Create graph datasets - PRESSURE
# ============================================================

train_dataset_pressure = data.create_graph_dataset(
    X_train_pressure,
    Y_train,
    edge_index,
    edge_weight
)

val_dataset_pressure = data.create_graph_dataset(
    X_val_pressure,
    Y_val,
    edge_index,
    edge_weight
)

test_dataset_pressure = data.create_graph_dataset(
    X_test_pressure,
    Y_test,
    edge_index,
    edge_weight
)


# ============================================================
# Dataset checks
# ============================================================

print(
    "Train velocity dataset:",
    len(train_dataset_velocity)
)

print(
    "Val velocity dataset:",
    len(val_dataset_velocity)
)

print(
    "Test velocity dataset:",
    len(test_dataset_velocity)
)

print(
    "Train pressure dataset:",
    len(train_dataset_pressure)
)

print(
    "Val pressure dataset:",
    len(val_dataset_pressure)
)

print(
    "Test pressure dataset:",
    len(test_dataset_pressure)
)

print(
    "Velocity input features:",
    X_train_velocity.shape[-1]
)

print(
    "Pressure input features:",
    X_train_pressure.shape[-1]
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
        self.dropout = nn.Dropout(
            dropout
        )

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
# Hyperparameters
# ============================================================

num_neurons = 64

num_layers = 3

learning_rate = 1e-3

batch_size = 8

dropout = 0

num_epochs = 200

weight_decay = 1e-5


# ============================================================
# Loss function
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# DataLoaders - VELOCITY
# ============================================================

train_loader_velocity = DataLoader(
    train_dataset_velocity,
    batch_size=batch_size,
    shuffle=True
)

val_loader_velocity = DataLoader(
    val_dataset_velocity,
    batch_size=batch_size,
    shuffle=False
)

test_loader_velocity = DataLoader(
    test_dataset_velocity,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# DataLoaders - PRESSURE
# ============================================================

train_loader_pressure = DataLoader(
    train_dataset_pressure,
    batch_size=batch_size,
    shuffle=True
)

val_loader_pressure = DataLoader(
    val_dataset_pressure,
    batch_size=batch_size,
    shuffle=False
)

test_loader_pressure = DataLoader(
    test_dataset_pressure,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# Create velocity network
#
# Input:
# [u_norm, v_norm, p_norm, delta_t]
#
# Output:
# [u, v]
# ============================================================

velocity_net = GCNNet(
    num_in=4,
    num_out=2,
    num_neurons=num_neurons,
    num_layers=num_layers,
    dropout=dropout
).to(device)


velocity_optimizer = optim.Adam(
    velocity_net.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)


# ============================================================
# Create pressure network
#
# Input:
# [
#   u_norm,
#   v_norm,
#   p_norm,
#   delta_t,
#   Fourier features
# ]
#
# Output:
# [p]
# ============================================================
num_neurons = 128

num_layers = 25

learning_rate = 1e-3

batch_size = 8

dropout = 0

num_epochs = 200

weight_decay = 1e-3


pressure_net = GCNNet(
    num_in=12,
    num_out=1,
    num_neurons=num_neurons,
    num_layers=num_layers,
    dropout=dropout
).to(device)


pressure_optimizer = optim.Adam(
    pressure_net.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)


# ============================================================
# Train velocity network
# ============================================================

print("\n============================================")
print("TRAINING VELOCITY NETWORK")
print("============================================")


(
    velocity_model_state,
    velocity_train_history,
    velocity_val_history
) = train_network(
    velocity_net,
    train_loader_velocity,
    val_loader_velocity,
    criterion,
    velocity_optimizer,
    num_epochs,
    device,
    output_type="velocity"
)


# Load best velocity model
velocity_net.load_state_dict(
    velocity_model_state
)

velocity_net.eval()


print(
    "\nBest velocity validation MSE: "
    f"{min(velocity_val_history):.6e}"
)


# ============================================================
# Train pressure network
# ============================================================

print("\n============================================")
print("TRAINING PRESSURE NETWORK")
print("============================================")


(
    pressure_model_state,
    pressure_train_history,
    pressure_val_history
) = train_network(
    pressure_net,
    train_loader_pressure,
    val_loader_pressure,
    criterion,
    pressure_optimizer,
    num_epochs,
    device,
    output_type="pressure"
)


# Load best pressure model
pressure_net.load_state_dict(
    pressure_model_state
)

pressure_net.eval()


print(
    "\nBest pressure validation MSE: "
    f"{min(pressure_val_history):.6e}"
)


# ============================================================
# Save best models
# ============================================================

torch.save(
    velocity_model_state,
    save_velocity_model
)

torch.save(
    pressure_model_state,
    save_pressure_model
)


print(
    f"\nBest velocity model saved at: "
    f"{save_velocity_model}"
)

print(
    f"Best pressure model saved at: "
    f"{save_pressure_model}"
)


# ============================================================
# Plot training histories
# ============================================================

print("\nVelocity training history")

plot.plot_loss(
    velocity_train_history,
    velocity_val_history
)


print("\nPressure training history")

plot.plot_loss(
    pressure_train_history,
    pressure_val_history
)


# ============================================================
# Test evaluation - VELOCITY
# ============================================================

velocity_test_losses = []

u_test_losses = []
v_test_losses = []


velocity_net.eval()


with torch.no_grad():

    for batch in test_loader_velocity:

        batch = batch.to(device)

        # Velocity prediction
        velocity_preds = velocity_net(
            batch
        )

        velocity_target = (
            batch.y[:, 0:2]
        )

        # Combined velocity MSE
        velocity_loss = criterion(
            velocity_preds,
            velocity_target
        )

        velocity_test_losses.append(
            velocity_loss.item()
        )

        # u MSE
        u_loss = criterion(
            velocity_preds[:, 0],
            batch.y[:, 0]
        )

        u_test_losses.append(
            u_loss.item()
        )

        # v MSE
        v_loss = criterion(
            velocity_preds[:, 1],
            batch.y[:, 1]
        )

        v_test_losses.append(
            v_loss.item()
        )


# ============================================================
# Test evaluation - PRESSURE
# ============================================================

pressure_test_losses = []

pressure_net.eval()


with torch.no_grad():

    for batch in test_loader_pressure:

        batch = batch.to(device)

        # Pressure prediction
        pressure_preds = pressure_net(
            batch
        )

        pressure_target = (
            batch.y[:, 2:3]
        )

        pressure_loss = criterion(
            pressure_preds,
            pressure_target
        )

        pressure_test_losses.append(
            pressure_loss.item()
        )


# ============================================================
# Mean test errors
# ============================================================

velocity_test_mse = np.mean(
    velocity_test_losses
)

u_test_mse = np.mean(
    u_test_losses
)

v_test_mse = np.mean(
    v_test_losses
)

pressure_test_mse = np.mean(
    pressure_test_losses
)


print("\n============================================")
print("TEST RESULTS")
print("============================================")

print(
    f"Velocity Test MSE: "
    f"{velocity_test_mse:.6e}"
)

print(
    f"u Test MSE:        "
    f"{u_test_mse:.6e}"
)

print(
    f"v Test MSE:        "
    f"{v_test_mse:.6e}"
)

print(
    f"Pressure Test MSE: "
    f"{pressure_test_mse:.6e}"
)


# ============================================================
# Select timestep for inference
# ============================================================

timestep = 25

X_current = X_input[timestep]

Y_true = Y_target[timestep]

delta_t_current = delta_t[timestep]


# ============================================================
# Predict next timestep
# ============================================================


Y_pred = inference.predict_next_timestep(
    velocity_model=velocity_net,
    pressure_model=pressure_net,
    X=X_current,
    delta_t=delta_t_current,
    edge_index=edge_index,
    edge_weight=edge_weight,
    x_mean=x_mean,
    x_std=x_std,
    dt_mean=dt_mean,
    dt_std=dt_std,
    device=device
)

