# Import necessary libraries
import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cat Sound Mood Classifier")

    # Add arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing mood folders with audio files",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/classifier.pkl"),
        help="Path to save/load the trained model",
    )
    parser.add_argument("--train", action="store_true", help="Train a new classifier")
    parser.add_argument(
        "--predict", type=Path, default=None, help="Audio file to predict mood for"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply audio augmentation during training",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    return parser.parse_args()
