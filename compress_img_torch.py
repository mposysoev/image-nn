"""
This code creates a Dense neural net that learns to reproduce input images.
The result improves with the increase in the number of layers and neurons.

Author: Maksim Posysoev
Date: 25 June 2024
License: GPL-3.0
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def load_image(image_path):
    """
    Load an image from the specified path and convert it to RGB format.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image in RGB format.

    Raises:
        ValueError: If the image is not found at the specified path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def prepare_data(image):
    """
    Prepare the image data for training by extracting coordinates and color values.

    Args:
        image (np.ndarray): The image data.

    Returns:
        tuple: Normalized coordinates, normalized colors, image height, and image width.
    """
    height, width, _ = image.shape
    coordinates = []
    colors = []

    for y in range(height):
        for x in range(width):
            coordinates.append([x, y])
            colors.append(image[y, x] / 255.0)

    coordinates = np.array(coordinates, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)

    coordinates[:, 0] /= width
    coordinates[:, 1] /= height

    return coordinates, colors, height, width


class DenseModel(nn.Module):
    def __init__(self):
        super(DenseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.model(x)


def train_model(model, dataloader, device, learning_rate, epochs, checkpoint_path):
    """
    Train the model with the given data and save checkpoints.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        device (torch.device): Device to run the training on.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        checkpoint_path (str): Base path for saving model checkpoints.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
        threshold=1e-4,
        threshold_mode="abs",
        min_lr=1e-7,
    )

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss}")
        print(f"Lr = {scheduler.get_last_lr()}")

        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                model.state_dict(), f"{checkpoint_path}_model_epoch_{epoch:02d}.pth"
            )

    torch.save(model.state_dict(), f"{checkpoint_path}_model_final.pth")


def main(image_path, epochs, batch_size, learning_rate, model_path=None):
    """
    Main function to load data, build or load model, and start training.

    Args:
        image_path (str): Path to the input image.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for training.
        model_path (str, optional): Path to an existing model to continue training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = load_image(image_path)
    X, Y, height, width = prepare_data(image)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model_path:
        model = DenseModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        model = DenseModel()
        print("Built a new model")

    train_model(model, dataloader, device, learning_rate, epochs, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network to compress an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument(
        "--model_path", type=str, help="Path to an existing model to continue training."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for training."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for training.",
    )
    args = parser.parse_args()

    main(
        args.image_path,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.model_path,
    )
