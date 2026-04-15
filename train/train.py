#!/usr/bin/env python3
#
#
# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision

from dataset_generator import (
    DataLoader,
    DatasetGenerator,
    DynamicGenerator,
    ImageLoader,
    load_paths,
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

torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)  < nll_loss2d_forward_out_cuda_template :/
torch.manual_seed(3)
# Currently, tile size needs to be smaller than the overlay tiles.
# tile_size = (400, 400)
tile_size = (256, 256)
alpha_factor = 0.5

# training_set = []
validation_set = []

from dataset_generator import DataPipeline

config_file = "dataset.priv.yaml"
train_pipeline = DataPipeline(config_file=config_file, full_init=False)
rng = np.random.default_rng(3)

validation_pipeline = train_pipeline.split_validation(rng=rng, ratio=0.1)
train_pipeline.post_image_init()


validation_set = [train_pipeline.generate(rng) for _ in range(100)]
validation_set = [(a.to(device), b.to(device)) for a, b in validation_set]


if True:
    for i, (img, mask) in enumerate(validation_set):
        epoch_dir = Path("/tmp/train/")
        epoch_dir.mkdir(exist_ok=True, parents=True)
        out_path = epoch_dir / f"validation_{i:0>3}.png"
        torchvision.utils.save_image([img, torch.stack([mask, mask, mask])], out_path)


resolution = tuple(validation_set[0][0].shape)
# Larger batches (no change to learning rate) is not actually better?
batch_size = 4

# Create data loaders for our datasets; shuffle for training, not for validation
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=batch_size, shuffle=False
)


batch_count = 40

dynamic_training_gen = DynamicGenerator(
    batch_generator=train_pipeline.batch_generator_fun(rng),
    batch_count=batch_count,
    batch_size=batch_size,
    device=device,
)


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
epoch_number = 0

EPOCHS = 10000

best_vloss = 1_000_000.0
model_seed = 3
torch.manual_seed(model_seed)
model = Unet(channels_in=3, channels_out=2)
model.to(device)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# learning_rate = 0.001  # for batch size of 4. Works well, plateau at 300, but saw one spike.
# learning_rate = 0.001  # for batch size of 4.
learning_rate = 0.00025  # Lowering it because data is more complex now.
# learning_rate = 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[25, 50, 300, 800],
    gamma=0.2,
)


loss_fn = torch.nn.CrossEntropyLoss()


def dump_stats(dir: Path, stats: dict):
    with open(dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def train_one_epoch(epoch_index):
    epoch_loss = 0.0
    count = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    with torch.no_grad():
        train_data = iter(dynamic_training_gen)
    for i, data in enumerate(train_data):
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
        loss.backward()
        print("batch loss", float(loss.detach()))
        epoch_loss += float(loss.detach())

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
    scheduler.step()
    return epoch_loss / count


stats = []
save_model = True
start_time = time.time()
for epoch in range(EPOCHS):
    epoch_record = {}
    print("EPOCH {}:".format(epoch))
    epoch_record["epoch"] = epoch

    start_train = time.time()
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(
        epoch,
    )
    end_train = time.time()
    epoch_record["train_loss"] = avg_loss
    epoch_record["train_time"] = end_train - start_train

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    epoch_dir = Path(f"/tmp/train/{epoch:0>3}/")

    if epoch > 100:
        epoch_dir = Path("/tmp/train/latest/")

    start_validation = time.time()
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

    end_validation = time.time()
    avg_vloss = running_vloss / (i + 1)
    epoch_record["validation_time"] = end_validation - start_validation
    epoch_record["validation_loss"] = float(avg_vloss.detach())
    elapsed_time = float(time.time() - start_time)
    epoch_record["elapsed_time"] = elapsed_time
    print(f"LOSS train {avg_loss} valid {avg_vloss}")
    print(
        "Train: {:.3} s validation: {:.3} s total: {:.3} s  elapsed {:.3}".format(
            end_train - start_train,
            end_validation - start_validation,
            end_validation - start_train,
            elapsed_time,
        )
    )

    stats.append(epoch_record)

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss and save_model:
        epoch_dir.mkdir(parents=True, exist_ok=True)
        best_vloss = avg_vloss
        # model_path = "/tmp/model_{}_{}".format(timestamp, epoch_number)
        model_path = epoch_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"saved model to {model_path}")

    dump_stats(epoch_dir, stats)
    dump_stats(epoch_dir.parent, stats)
