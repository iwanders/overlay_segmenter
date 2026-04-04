#!/usr/bin/env python3
#
#
# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision

from dataset_generator import (
    DatasetGenerator,
    DynamicGenerator,
    ImageLoader,
    rng_shuffle,
)
from model import Unet

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = torch.device("cuda:0")  # or "cuda" for the current device
else:
    print("No GPU available. Training will run on CPU.")
    device = torch.device("cpu")

print(f"Using device: {device}")


# Currently, tile size needs to be smaller than the overlay tiles.
# tile_size = (400, 400)
tile_size = (256, 256)
alpha_factor = 0.5

# training_set = []
validation_set = []

training_generators = []


background_dir = Path("../../datasets/background/")
foreground_dir = Path("../../datasets/foreground/")

background_images = []
foreground_images = []

for tileset in ["cave", "barracks"]:
    background_images.extend(
        ImageLoader.background_loader(background_dir / tileset).images()
    )
    foreground_images.extend(
        ImageLoader.foreground_loader(foreground_dir / tileset).images()
    )


rng = np.random.default_rng(seed=3)
background_images = rng_shuffle(rng, background_images)
foreground_images = rng_shuffle(rng, foreground_images)


total_bg = len(background_images)
validation_bg_split = int(total_bg * 0.1)
validation_bg = background_images[0:validation_bg_split]
train_bg = background_images[validation_bg_split + 1 :]
print(f"Validation bgs: {len(validation_bg)}")
print(f"Train bgs: {len(train_bg)}")

validation_set = []
validation_generator = DatasetGenerator(
    background_images=validation_bg,
    foreground_images=foreground_images,
    device=device,
    rng=rng,
    tile_size=tile_size,
    alpha_factor=alpha_factor,
)
validation_set.extend(
    validation_generator.generate(
        count=30,
    )
)


if True:
    for i, (img, mask) in enumerate(validation_set):
        epoch_dir = Path("/tmp/train/")
        epoch_dir.mkdir(exist_ok=True, parents=True)
        out_path = epoch_dir / f"validation_{i:0>3}.png"
        torchvision.utils.save_image([img, torch.stack([mask, mask, mask])], out_path)


BATCH_SIZE_LOOKUP = {
    (3, 256, 256): 12,  # 6.9 GB of vram
}
resolution = tuple(validation_set[0][0].shape)
batch_size = BATCH_SIZE_LOOKUP.get(resolution, 4)
# Larger batches (no change to learning rate) is not actually better?
batch_size = 4

# Create data loaders for our datasets; shuffle for training, not for validation
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=batch_size, shuffle=False
)

training_gen = DatasetGenerator(
    background_images=train_bg,
    foreground_images=foreground_images,
    device=device,
    alpha_factor=alpha_factor,
    rng=rng,
    tile_size=tile_size,
)

batch_count = 20

dynamic_training_gen = DynamicGenerator(
    training_gen, batch_count=batch_count, batch_size=batch_size
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
learning_rate = 0.001  # for batch size of 4.
# learning_rate = 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


loss_fn = torch.nn.CrossEntropyLoss()


def train_one_epoch(epoch_index):
    epoch_loss = 0.0
    count = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(dynamic_training_gen):
        # for i, data in enumerate(training_loader):
        # print("data", type(data))
        # Every data instance is an input + label pair
        inputs, labels = data
        # print("inputs", type(inputs), inputs.shape)
        # print("labels", type(labels), labels.shape)

        count += 1

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        # print("outputs", outputs.shape, outputs.dtype)
        # print("labels", labels.shape, labels.dtype)

        # labels = labels.softmax(dim=1)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        print("batch loss", float(loss))
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


save_model = True
for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(
        epoch_number,
    )

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    epoch_dir = Path(f"/tmp/train/{epoch:0>3}/")
    epoch_dir = Path(f"/tmp/train/{epoch:0>3}/")
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

            # And lets write that to disk shall we.
            batch_size = vinputs.shape[0]
            if True:  # and (epoch < 10 or epoch % 10 == 0):
                epoch_dir.mkdir(parents=True, exist_ok=True)
                for frame_i in range(batch_size):
                    real_i = i * batch_size + frame_i
                    this_slice = voutputs[frame_i, :, :]
                    this_target = vlabels[frame_i, :, :]
                    """
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
                    """
                    mask_img = epoch_dir / f"eval_{real_i:0>5}_mask.png"
                    index_mask = this_slice.argmax(0)
                    torchvision.utils.save_image(index_mask.to(torch.float), mask_img)
                    # print(f"index_mask: {index_mask.shape}", index_mask)
                    values_img = epoch_dir / f"eval_{real_i:0>5}_values.png"
                    t = this_slice[1, :, :]
                    span = t.max() - t.min()
                    t = (t - t.min()) / span
                    torchvision.utils.save_image(t.to(torch.float), values_img)
                    target_img = epoch_dir / f"eval_{real_i:0>5}_target.png"
                    torchvision.utils.save_image(
                        this_target.to(torch.float), target_img
                    )
                    if epoch < 1000:
                        image_img = epoch_dir / f"eval_{real_i:0>5}_image.png"
                        torchvision.utils.save_image(vinputs[frame_i, :, :], image_img)

    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss and save_model:
        epoch_dir.mkdir(parents=True, exist_ok=True)
        best_vloss = avg_vloss
        # model_path = "/tmp/model_{}_{}".format(timestamp, epoch_number)
        model_path = epoch_dir / "model.pth"
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
