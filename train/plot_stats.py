#!/usr/bin/env python3
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

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

    epoch_interval = np.array(epoch)
    time_interval = np.array([x["elapsed_time"] for x in d])

    def forward(epoch):
        return np.interp(epoch, epoch_interval, time_interval)

    def inverse(time):
        return np.interp(time, time_interval, epoch_interval)

    secax = plt.gca().secondary_xaxis("top", functions=(forward, inverse))
    secax.set_xlabel("train time [s]")
    plt.legend()  # Displays the labels
    # plt.show()
    plt.savefig("/tmp/stats.svg")
