import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor


def load_image_file(d, device):
    image = Image.open(d)
    image = ToTensor()(image)
    image = image.to(device)
    # print("load image", type(image))
    return image


def load_image_file_u8(d, device):
    image = Image.open(d)
    image = torchvision.transforms.functional.pil_to_tensor(image)
    image = image.to(device)
    return image


def determine_prefered_device() -> str:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        return torch.device("cuda:0")  # or "cuda" for the current device
    else:
        print("No GPU available. Training will run on CPU.")
        return torch.device("cpu")


preferred_device = determine_prefered_device()


def lookup_device(input_str) -> torch.device:
    if input_str == "auto":
        return preferred_device

    return input_str
