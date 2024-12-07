from PIL import Image
import json
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def random_pick_two_elements(elements):
    sampled_elements = random.sample(elements + elements, 2)
    return sampled_elements[0], sampled_elements[1]


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])