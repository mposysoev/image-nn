"""
This code read *.pth file and reproduce the image that stored there.

Author: Maksim Posysoev
Date: 25 June 2024
License: GPL-3.0
"""

import argparse
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def load_model(model_path):
    """Load a PyTorch model from the given path."""
    try:
        model = DenseModel()
        model_state = torch.load(model_path)
        model.load_state_dict(model_state)
        model.eval()
        logging.info(f"Model loaded successfully from {model_path}.")
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise ValueError(f"Failed to load model from {model_path}: {e}")
    return model


def generate_normalized_coordinates(height, width):
    """Generate normalized coordinates for an image of given height and width."""
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    coordinates = np.stack([x_coords / width, y_coords / height], axis=-1)
    return coordinates.reshape(-1, 2).astype(np.float32)


def generate_normalized_coordinates_extended(
    original_height, original_width, target_height, target_width
):
    """Generate normalized coordinates for an extended image with padding."""
    y_offset = (target_height - original_height) // 2
    x_offset = (target_width - original_width) // 2

    y_coords, x_coords = np.meshgrid(
        np.arange(target_height), np.arange(target_width), indexing="ij"
    )
    norm_x = (x_coords - x_offset) / original_width
    norm_y = (y_coords - y_offset) / original_height
    coordinates = np.stack([norm_x, norm_y], axis=-1)
    return coordinates.reshape(-1, 2).astype(np.float32)


def restore_image(model, height, width, device):
    """Restore an image using the model for the given height and width."""
    coordinates = generate_normalized_coordinates(height, width)
    coordinates_tensor = torch.tensor(coordinates).to(device)
    with torch.no_grad():
        predicted_colors = model(coordinates_tensor).cpu().numpy()
    predicted_colors = predicted_colors.reshape((height, width, 3))
    return np.clip(predicted_colors * 255, 0, 255).astype("uint8")


def restore_image_with_extension(
    model, original_height, original_width, target_height, target_width, device
):
    """Restore an extended image using the model."""
    coordinates = generate_normalized_coordinates_extended(
        original_height, original_width, target_height, target_width
    )
    coordinates_tensor = torch.tensor(coordinates).to(device)
    with torch.no_grad():
        predicted_colors = model(coordinates_tensor).cpu().numpy()
    predicted_colors = predicted_colors.reshape((target_height, target_width, 3))
    return np.clip(predicted_colors * 255, 0, 255).astype("uint8")


def save_image(image, output_path):
    """Save the image to the specified path."""
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logging.info(f"Restored image saved to {output_path}.")


def validate_arguments(height, width, target_height, target_width):
    """Validate the provided arguments for height and width."""
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive integers.")
    if (target_height is not None and target_height <= 0) or (
        target_width is not None and target_width <= 0
    ):
        raise ValueError(
            "Target height and width must be positive integers if provided."
        )


def main(model_path, height, width, target_height=None, target_width=None):
    """Main function to restore and optionally extend an image."""
    validate_arguments(height, width, target_height, target_width)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path).to(device)

    if target_height is None or target_width is None:
        restored_image = restore_image(model, height, width, device)
        output_path = f"{model_path}_restored.png"
    else:
        restored_image = restore_image_with_extension(
            model, height, width, target_height, target_width, device
        )
        output_path = f"{model_path}_restored_{target_height}x{target_width}.png"

    save_image(restored_image, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restore and optionally expand an image using a trained neural network model."
    )
    parser.add_argument("model_path", type=str, help="Path to the trained model file.")
    parser.add_argument("height", type=int, help="Original height of the image.")
    parser.add_argument("width", type=int, help="Original width of the image.")
    parser.add_argument(
        "--target_height", type=int, help="Target height of the restored image."
    )
    parser.add_argument(
        "--target_width", type=int, help="Target width of the restored image."
    )
    args = parser.parse_args()

    main(
        args.model_path, args.height, args.width, args.target_height, args.target_width
    )
