import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb

# Initialize wandb
wandb.init(project="MLX_MNIST")


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        return self.layers[-1](x)


def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")


def main():
    seed = 0
    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1

    np.random.seed(seed)

    # Data preprocessing and loading with input flattening
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten input tensor
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = MLP(num_layers, 28 * 28, hidden_dim, num_classes)

    optimizer = optim.SGD(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Log the hyperparameters with wandb
    wandb.config = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X, y in train_loader:
            X, y = mx.array(X.numpy()), mx.array(y.numpy())
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            epoch_loss += loss.item()  # Accumulate loss

        # Evaluation
        test_images, test_labels = next(iter(test_loader))
        test_images, test_labels = (
            mx.array(test_images.numpy()),
            mx.array(test_labels.numpy()),
        )
        accuracy = mx.mean(mx.argmax(model(test_images), axis=1) == test_labels)

        # Log metrics with wandb
        wandb.log(
            {
                "epoch": epoch,
                "loss": epoch_loss / len(train_loader),  # Average training loss
                "accuracy": accuracy.item(),
            }
        )

        print(
            f"Epoch {epoch}: Test accuracy {accuracy.item():.3f}, Avg Loss: {epoch_loss / len(train_loader):.3f}"
        )


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    main()
