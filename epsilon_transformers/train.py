#!/usr/bin/env python
import argparse
from epsilon_transformers.training.trainer import train_model
from epsilon_transformers.training.configs import TrainConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "rnn"], required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    # Add other CLI arguments
    
    args = parser.parse_args()
    config = TrainConfig.from_args(args)
    train_model(config)

if __name__ == "__main__":
    main()