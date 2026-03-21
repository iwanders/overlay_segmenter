#!/usr/bin/env python3
#
#
# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
from datetime import datetime
from pathlib import Path

import torch
import torchvision

from drive_loader import load_drive_dataset
from model import Unet

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = torch.device("cuda:0")  # or "cuda" for the current device
else:
    print("No GPU available. Training will run on CPU.")
    device = torch.device("cpu")

print(f"Using device: {device}")

train, test = load_drive_dataset(device=device)

training_set = [(a.image, a.manual1) for a in train][0:3]
validation_set = [(a.image, a.manual1) for a in train + test][0:3]


# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=4, shuffle=False
)


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
epoch_number = 0

EPOCHS = 100

best_vloss = 1_000_000.0

model = Unet(channels_in=3, channels_out=2)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

loss_fn = torch.nn.CrossEntropyLoss()


def train_one_epoch(epoch_index):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        print("outputs", outputs.shape, outputs.dtype)
        print("labels", labels.shape, labels.dtype)

        # labels = labels.softmax(dim=1)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        print(loss)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        """
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999 or True:
            last_loss = running_loss / 1000  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0
        """

    return last_loss


for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    epoch_dir = Path(f"/tmp/train/{epoch:0>3}/")
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(
        epoch_number,
    )

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

            # And lets write that to disk shall we.
            batch_size = vinputs.shape[0]
            for frame_i in range(batch_size):
                real_i = i * batch_size + frame_i
                mask_img = epoch_dir / f"eval_{real_i:0>5}_mask.png"
                torchvision.utils.save_image(
                    voutputs[frame_i, :, :][0, :, :].softmax(0), mask_img
                )
                image_img = epoch_dir / f"eval_{real_i:0>5}_image.png"
                torchvision.utils.save_image(vinputs[frame_i, :, :], image_img)

    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        # model_path = "/tmp/model_{}_{}".format(timestamp, epoch_number)
        model_path = epoch_dir / "model.pth"
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
