#!/usr/bin/env python3
import json
import sys

import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        d = json.load(f)

    epoch = [x["epoch"] for x in d]
    train_loss = [x["train_loss"] for x in d]
    validation_loss = [x["validation_loss"] for x in d]

    plt.plot(epoch, train_loss, label="Train", color="blue", linestyle="--")
    plt.plot(epoch, validation_loss, label="Validation", color="red", marker="")

    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.legend()  # Displays the labels
    # plt.show()
    plt.savefig("/tmp/stats.svg")
