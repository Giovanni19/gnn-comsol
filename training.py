import numpy as np
import torch
import copy


def train_network(
    net,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device
):
    """
    Train the GCN and select the model with the lowest validation loss.

    Parameters
    ----------
    net : torch.nn.Module
        Neural network to train.

    train_loader : DataLoader
        Training data loader.

    val_loader : DataLoader
        Validation data loader.

    criterion : torch.nn.Module
        Loss function.

    optimizer : torch.optim.Optimizer
        Optimizer.

    num_epochs : int
        Number of training epochs.

    device : torch.device
        Device used for training (CPU or GPU).

    Returns
    -------
    best_model_state : dict
        State dictionary of the model with the lowest validation loss.

    train_loss_history : list
        Training loss for each epoch.

    val_loss_history : list
        Validation loss for each epoch.
    """

    # Best validation loss
    best_val_loss = np.inf

    # Parameters of the best model
    best_model_state = None

    # Loss history
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):

        # ==========================================
        # Training
        # ==========================================

        net.train()

        train_losses = []

        for batch in train_loader:

            # Move graph batch to device
            batch = batch.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            preds = net(batch)

            # Compute loss
            loss = criterion(preds, batch.y)

            # Backpropagation
            loss.backward()

            # Update network parameters
            optimizer.step()

            # Store batch loss
            train_losses.append(loss.item())

        # Mean training loss
        mean_train_loss = np.mean(train_losses)


        # ==========================================
        # Validation
        # ==========================================

        net.eval()

        val_losses = []

        with torch.no_grad():

            for batch in val_loader:

                # Move graph batch to device
                batch = batch.to(device)

                # Forward pass
                preds = net(batch)

                # Compute validation loss
                loss = criterion(preds, batch.y)

                # Store batch loss
                val_losses.append(loss.item())

        # Mean validation loss
        mean_val_loss = np.mean(val_losses)


        # ==========================================
        # Store history
        # ==========================================

        train_loss_history.append(mean_train_loss)
        val_loss_history.append(mean_val_loss)


        # ==========================================
        # Print epoch results
        # ==========================================

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {mean_train_loss:.6e} | "
            f"Val Loss: {mean_val_loss:.6e}"
        )


        # ==========================================
        # Save best model
        # ==========================================

        if mean_val_loss < best_val_loss:

            best_val_loss = mean_val_loss

            best_model_state = copy.deepcopy(
                net.state_dict()
            )


    return (
        best_model_state,
        train_loss_history,
        val_loss_history
    )