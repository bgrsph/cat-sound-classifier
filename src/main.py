"""
Cat Sound Mood Classifier - CLI argument parsing.
"""

import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cat Sound Mood Classifier")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data-dir", type=Path, default=Path("data/interim"))
    train_parser.add_argument("--model-path", type=Path, default=Path("models/cat_meow.pt"))
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=0.001)
    
    # Test
    test_parser = subparsers.add_parser("test", help="Evaluate a model")
    test_parser.add_argument("--model-path", type=Path, required=True)
    test_parser.add_argument("--data-dir", type=Path, default=Path("data/interim"))
    test_parser.add_argument("--batch-size", type=int, default=16)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
