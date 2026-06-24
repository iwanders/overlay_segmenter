# Overlay Segmenter

## Inference
Use [overlay_segmenter](./overlay_segmenter) to run the network, should be pretty straightforward to integrate.


## Training

Done with Python, using the [./train](./train) directory, an example;
```
./train.py ./dataset_example.yaml
```

Resume training from a checkpoint with:
```
./train.py  -l /tmp/train/latest/model.pth
```

Inference using a checkpoint, writes files adjacent to the images for inspection.
```
./inference.py inference -c <checkpoint> <images>
```

Convert checkpoint to safetensors for usage from Rust, can also do f16 conversion etc.
```
./inference.py convert -c <checkpoint> --output /tmp/unet.safetensors
```
