from psd_tools import PSDImage
from psd_tools.api.mask import Mask
import numpy as np
from PIL import Image


def create_psd_from_image(image):
    psd = PSDImage.new(image.size[0], image.size[1])
    psd.add_layer(image)
    return psd


def is_in_image(point, size):
    w = size[0]
    h = size[1]
    is_in = True
    if point[0] < 0 or point[0] > w or point[1] < 0 or point[1] > h:
        is_in = False

    return is_in


def mask_to_layer(image, mask, mode):
    if mode == "cut":
        array = np.array(image)
        print(array.shape)
        mask = mask.reshape(1, mask.shape[0], mask.shape[1])
        layer = array * mask
        layer = Image.fromarray(layer)
        out_mask = None

    elif mode == "mask only":
        mask = (mask * 255).astype(np.uint8)
        layer = Image.fromarray(mask)
        out_mask = None

    elif mode == "both":
        layer = image
        mask = (mask * 255).astype(np.uint8)
        out_mask = Image.fromarray(mask)
    else:
        print("Invalid mode.")
        layer = None
        out_mask = None

    return layer, out_mask


def add_layer_and_mask(psd: PSDImage, layer:Image, mask: Image):
    if layer is not None:
        source_layer = psd.add_layer(layer)
        if mask is not None:
            mask = Mask(mask.width, mask.height, mask.tobytes())
            source_layer.mask = mask

    return psd
