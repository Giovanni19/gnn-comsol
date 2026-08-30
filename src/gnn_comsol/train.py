"""
The training loop.

There used to be three near-identical copies of this file, differing only
in how they picked the target columns, and the `output_type` argument
meant something different in each. There is now one loop, and the target
selection is a single explicit argument.
"""

import copy

import numpy as np
import torch


def train_network(
    net,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    target_columns=slice(None),
    verbose=True
):
    """
    Train a network and keep the weights with the lowest validation loss.

    Parameters
    ----------
    target_columns : slice
        Which columns of batch.y this network is responsible for.
        VELOCITY_COLUMNS, PRESSURE_COLUMNS or STATE_COLUMNS from
        gnn_comsol.data.normalization. The datasets always carry the full
        normalized state as the target, so one dataset can serve networks
        predicting different variables.

    Returns
    -------
    best_model_state : dict
        Weights at the epoch with the lowest validation loss.

    train_loss_history, val_loss_history : list of float

    Note: this keeps the best checkpoint but does not stop early; the
    loop always runs for num_epochs.
    """

    best_val_loss = np.inf
    best_model_state = None

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):

        # ---------------------------------------------------
        # Training
        # ---------------------------------------------------

        net.train()

        train_losses = []

        for batch in train_loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            preds = net(batch)
            target = batch.y[:, target_columns]

            loss = criterion(preds, target)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))

        # ---------------------------------------------------
        # Validation
        # ---------------------------------------------------

        net.eval()

        val_losses = []

        with torch.no_grad():

            for batch in val_loader:

                batch = batch.to(device)

                preds = net(batch)
                target = batch.y[:, target_columns]

                val_losses.append(
                    criterion(preds, target).item()
                )

        mean_val_loss = float(np.mean(val_losses))

        train_loss_history.append(mean_train_loss)
        val_loss_history.append(mean_val_loss)

        if verbose:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {mean_train_loss:.6e} | "
                f"Val Loss: {mean_val_loss:.6e}"
            )

        if mean_val_loss < best_val_loss:

            best_val_loss = mean_val_loss
            best_model_state = copy.deepcopy(net.state_dict())

    return best_model_state, train_loss_history, val_loss_history
