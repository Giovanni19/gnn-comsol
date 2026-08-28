import h5py
import numpy as np
import torch
from torch_geometric.data import Data


def load_data(file_path):
    with h5py.File(file_path, "r") as f:
        X = np.array(f["X"])
        edge_index = np.array(f["edge_index"])
        edge_weight = np.array(f["edge_weight"])
        P = np.array(f["P"])
        t = np.array(f["t"])
        h = np.array(f["h"])
    ## MATLAB saves the matrices in a different format
    X = np.transpose(X, (2, 1, 0))

    edge_weight = edge_weight.squeeze()
    t = t.squeeze() #suitable for numpy
    h = h.squeeze() #suitable for numpy


    # Exclude the first timestep, let's see if things improve
    X_input = X[1:-1]
    Y_target = X[2:]
    delta_t = t[1:] - t[:-1]

    return X_input, Y_target, edge_index, edge_weight, delta_t, h

    
# Aggiungiamo il timestep, sicuramente ci servira quando avremo tante simulazioni


def add_time_feature(X, time_values):

    num_timesteps = X.shape[0]
    num_nodes = X.shape[1]

    time_feature = np.zeros(
        (num_timesteps, num_nodes, 1)
    )

    for i in range(num_timesteps):
        time_feature[i, :, 0] = time_values[i]

    X_with_time = np.concatenate(
        (
            X,
            time_feature
        ),
        axis=2
    )

    return X_with_time


#Proviamo ad aggiungere i fourier features con il mio timestep

def add_time_fourier_features(X, delta_t, num_frequencies=4):

    # Number of snapshots and nodes
    n_snapshots = X.shape[0]
    n_nodes = X.shape[1]

    # Frequencies for Fourier features
    frequencies = 2.0 ** np.arange(num_frequencies)

    # List containing original features
    features = [X]

    # Add raw delta_t
    dt_feature = np.repeat(
        delta_t[:, None, None],
        n_nodes,
        axis=1
    )

    features.append(dt_feature)

    # Add Fourier features
    for frequency in frequencies:

        sin_feature = np.sin(
            2.0 * np.pi * frequency * delta_t
        )

        cos_feature = np.cos(
            2.0 * np.pi * frequency * delta_t
        )

        sin_feature = np.repeat(
            sin_feature[:, None, None],
            n_nodes,
            axis=1
        )

        cos_feature = np.repeat(
            cos_feature[:, None, None],
            n_nodes,
            axis=1
        )

        features.append(sin_feature)
        features.append(cos_feature)

    # Concatenate all features
    X_with_time = np.concatenate(
        features,
        axis=2
    )

    return X_with_time





# Proviamo ad aggiungere le derivate nel tempo come features

def add_time_derivative_features(
    X_norm,
    X_original,
    dt
):
    """
    Add non-normalized time derivative features to the normalized data.
    """

    # Compute time derivatives
    dX_dt = np.zeros_like(X_original)

    dX_dt[1:] = (
        X_original[1:] - X_original[:-1]
    ) / dt

    # For the first timestep, use the derivative
    # computed between the first two available timesteps
    dX_dt[0] = dX_dt[1]

    # Concatenate normalized physical variables
    # and non-normalized time derivatives
    X_with_derivatives = np.concatenate(
        (
            X_norm,
            dX_dt
        ),
        axis=2
    )

    return X_with_derivatives

def split_dataset(
    X_input,
    Y_target,
    delta_t,
    train_fraction=0.70,
    val_fraction=0.15,
    seed=68
):

    # Generate shuffled indices
    rng = np.random.default_rng(seed)

    indices = np.arange(len(X_input))
    rng.shuffle(indices)

    # Shuffle input, target and delta_t
    # using exactly the same indices
    X_input = X_input[indices]
    Y_target = Y_target[indices]
    delta_t = delta_t[indices]

    # Total number of samples
    n_samples = len(X_input)

    # Number of training and validation samples
    n_train = int(train_fraction * n_samples)
    n_val = int(val_fraction * n_samples)

    # Split dataset
    X_train = X_input[:n_train]
    Y_train = Y_target[:n_train]
    dt_train = delta_t[:n_train]

    X_val = X_input[n_train:n_train+n_val]
    Y_val = Y_target[n_train:n_train+n_val]
    dt_val = delta_t[n_train:n_train+n_val]

    X_test = X_input[n_train+n_val:]
    Y_test = Y_target[n_train+n_val:]
    dt_test = delta_t[n_train+n_val:]

    return (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        dt_train,
        dt_val,
        dt_test
    )






def compute_normalization_parameters(X_train, dt_train):
    # Velocity components
    u = X_train[:, :, 0]
    v = X_train[:, :, 1]

    # Velocity magnitude
    velocity_magnitude = np.sqrt(u**2 + v**2)

    # Mean and standard deviation of velocity magnitude
    velocity_mean = velocity_magnitude.mean()
    velocity_std = velocity_magnitude.std() + 1e-8

    # Pressure
    p = X_train[:, :, 2]

    # Mean and standard deviation of pressure
    pressure_mean = p.mean()
    pressure_std = p.std() + 1e-8

    # Timestep
    delta_t_mean = dt_train.mean()
    delta_t_std = dt_train.std() + 1e-8

    # Same normalization parameters for u and v
    x_mean = np.array([
        velocity_mean,
        velocity_mean,
        pressure_mean
    ])

    x_std = np.array([
        velocity_std,
        velocity_std,
        pressure_std
    ])

    return x_mean, x_std, delta_t_mean, delta_t_std






def normalize_datasets(
    X_train,
    Y_train,
    X_val,
    Y_val,
    X_test,
    Y_test,
    x_mean,
    x_std
):
    # Normalize input: u, v, p, delta_t
    X_train_norm = (X_train - x_mean) / x_std
    X_val_norm = (X_val - x_mean) / x_std
    X_test_norm = (X_test - x_mean) / x_std

    # Normalize target: u, v, p
    # delta_t is not part of the target
    y_mean = x_mean[:3]
    y_std = x_std[:3]

    Y_train_norm = (Y_train - y_mean) / y_std
    Y_val_norm = (Y_val - y_mean) / y_std
    Y_test_norm = (Y_test - y_mean) / y_std

    return (
        X_train_norm,
        X_val_norm,
        X_test_norm,
    )


#qui ho tolto il target normalizzato per vedere cosa succede

def convert_to_torch(
    X_train_norm,
    Y_train_norm,
    X_val_norm,
    Y_val_norm,
    X_test_norm,
    Y_test_norm,
    edge_index,
    edge_weight
):
    X_train = torch.tensor(X_train_norm, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_norm, dtype=torch.float32)

    X_val = torch.tensor(X_val_norm, dtype=torch.float32)
    Y_val = torch.tensor(Y_val_norm, dtype=torch.float32)

    X_test = torch.tensor(X_test_norm, dtype=torch.float32)
    Y_test = torch.tensor(Y_test_norm, dtype=torch.float32)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    return (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        edge_index,
        edge_weight
    )


def create_graph_dataset(X, Y, edge_index, edge_weight):

    dataset = []

    for i in range(X.shape[0]):

        data = Data(
            x=X[i],
            y=Y[i],
            edge_index=edge_index,
            edge_weight=edge_weight
        )

        dataset.append(data)

    return dataset