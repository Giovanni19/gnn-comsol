import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses):

    # Number of epochs
    num_epochs = len(train_losses)

    # Epoch vector
    epochs = range(1, num_epochs + 1)

    # Create figure
    plt.figure(figsize=(8, 5))

    # Plot losses
    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    # Labels and title
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")

    # Legend and grid
    plt.legend()
    plt.grid(True)

    # Improve layout
    plt.tight_layout()

    # Show figure
    plt.show()


