"""
This code create a Dense neural net that learn to reproduce input image.
Result the better the more amount of layers and neurons.

Author: Maksim Posysoev
Date: 25 June 2024
License: GPL-3.0
"""

import argparse
import cv2
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


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


def build_dense_model(input_shape, learning_rate):
    """
    Build and compile a dense neural network model.

    Args:
        input_shape (tuple): Shape of the input data.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.Model: Compiled model.
    """
    model = Sequential(
        [
            Dense(1024, activation="relu", input_shape=input_shape),
            Dense(512, activation="relu"),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(3, activation="linear"),
        ]
    )
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def train_model(model, X, Y, image_path, epochs, batch_size):
    """
    Train the model with the given data and save checkpoints.

    Args:
        model (tf.keras.Model): The model to train.
        X (np.ndarray): Input data.
        Y (np.ndarray): Target data.
        image_path (str): Base path for saving model checkpoints.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    checkpoint_callback = ModelCheckpoint(
        filepath=f"{image_path}_model_epoch_{{epoch:02d}}.keras",
        save_weights_only=False,
        save_best_only=False,
        mode="auto",
        save_freq="epoch",
    )

    early_stopping_callback = EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )

    reduce_lr_callback = ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
    )

    model.fit(
        X,
        Y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback],
    )

    model.save(f"{image_path}_model_final.keras")


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
    image = load_image(image_path)
    X, Y, height, width = prepare_data(image)

    if model_path:
        try:
            model = load_model(model_path)
            print(f"Loaded model from {model_path}")
        except (OSError, ValueError) as e:
            print(f"Error loading model from {model_path}: {e}")
            model = build_dense_model(input_shape=(2,), learning_rate=learning_rate)
            print("Built a new model instead")
    else:
        model = build_dense_model(
            input_shape=(2,),
            learning_rate=learning_rate,
        )
        print("Built a new model")

    train_model(model, X, Y, image_path, epochs, batch_size)


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
        default=0.0001,
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
