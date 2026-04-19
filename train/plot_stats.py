#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    stats_path = Path(sys.argv[1])
    with open(stats_path) as f:
        d = json.load(f)

    epoch = [x["epoch"] for x in d]
    train_loss = [x["train_loss"] for x in d]
    validation_loss = [x["validation_loss"] for x in d]
    lowest_validation = np.min(validation_loss)
    lowest_train = np.min(train_loss)
    # marker =  "" linestyle="-",
    plt.plot(epoch, train_loss, label="Train", color="blue", linewidth=0.5)
    plt.axhline(y=lowest_train, color="blue", linestyle="--", linewidth=0.5)
    plt.text(
        0,
        lowest_train,
        f"{lowest_train}",
        color="blue",
        va="top",
        # fontweight="bold",
    )
    plt.plot(
        epoch,
        validation_loss,
        label="Validation",
        color="red",
        linewidth=0.5,
    )
    plt.axhline(y=lowest_validation, color="red", linestyle="--", linewidth=0.5)
    plt.text(
        0,
        lowest_validation,
        f"{lowest_validation}",
        color="red",
        va="bottom",
        # fontweight="bold",
    )

    max_epoch = epoch[-1]
    max_t = d[-1]["elapsed_time"]
    plt.axvline(x=max_epoch, color="black", linestyle="--", linewidth=0.5)
    plt.text(
        max_epoch,
        0,
        f"epoch {max_epoch}, t: {max_t:.2f}",
        color="black",
        ha="right",
        va="bottom",
        # fontweight="bold",
    )

    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.ylim([0.0, 0.4])
    plt.title("Training Loss")
    plt.gcf().set_size_inches(8 * 2, 4 * 2)

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
    plt.savefig(stats_path.with_name("stats.svg"))
