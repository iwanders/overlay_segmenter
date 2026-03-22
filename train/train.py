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

if True:
    for i, v in enumerate(train):
        epoch_dir = Path("/tmp/train/")
        out_path = epoch_dir / f"{i:0>3}_train.png"
        torchvision.utils.save_image(
            [v.image, torch.stack([v.manual1, v.manual1, v.manual1])], out_path
        )

training_set = [(a.image, a.manual1) for a in train][0:1]
validation_set = [(a.image, a.manual1) for a in train + test][0:1]


# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=4, shuffle=False
)


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
epoch_number = 0

EPOCHS = 10000

best_vloss = 1_000_000.0

model = Unet(channels_in=3, channels_out=2)
model.to(device)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

loss_fn = torch.nn.CrossEntropyLoss()


def train_one_epoch(epoch_index):
    epoch_loss = 0.0
    count = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        count += 1

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        print("outputs", outputs.shape, outputs.dtype)
        print("labels", labels.shape, labels.dtype)

        # labels = labels.softmax(dim=1)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        print("batch loss", loss)
        loss.backward()
        epoch_loss += float(loss)

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

    return epoch_loss / count


save_model = False
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
            if True:
                for frame_i in range(batch_size):
                    real_i = i * batch_size + frame_i
                    this_slice = voutputs[frame_i, :, :]
                    print(f"outputs: {voutputs.shape}")
                    print(
                        f"this_slice: {this_slice.shape}, this_slice[0,:,:].min() and max",
                        this_slice[0, :, :].min(),
                        this_slice[0, :, :].max(),
                    )
                    print(
                        "                                       this_slice[1,:,:].min() and max",
                        this_slice[1, :, :].min(),
                        this_slice[1, :, :].max(),
                    )
                    mask_img = epoch_dir / f"eval_{real_i:0>5}_mask.png"
                    index_mask = this_slice.argmax(0)
                    torchvision.utils.save_image(index_mask.to(torch.float), mask_img)
                    # print(f"index_mask: {index_mask.shape}", index_mask)
                    values_img = epoch_dir / f"eval_{real_i:0>5}_values.png"
                    t = this_slice[0, :, :]
                    span = t.max() - t.min()
                    t = (t - t.min()) / span
                    print(t)
                    torchvision.utils.save_image(t.to(torch.float), values_img)
                    if epoch < 1000:
                        image_img = epoch_dir / f"eval_{real_i:0>5}_image.png"
                        torchvision.utils.save_image(vinputs[frame_i, :, :], image_img)

    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss and save_model:
        best_vloss = avg_vloss
        # model_path = "/tmp/model_{}_{}".format(timestamp, epoch_number)
        model_path = epoch_dir / "model.pth"
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
