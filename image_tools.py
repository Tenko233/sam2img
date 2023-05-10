import os
import numpy as np
from PIL import Image


def is_in_image(point, size):
    w = size[0]
    h = size[1]
    is_in = True
    if point[0] < 0 or point[0] > w or point[1] < 0 or point[1] > h:
        is_in = False

    return is_in


def mask_to_layer(image, mask, mode):
    if mode == "crop":
        array = np.array(image)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        layer = array * mask
        layer = Image.fromarray(layer)

    elif mode == "mask":
        mask = (mask * 255).astype(np.uint8)
        layer = Image.fromarray(mask)

    else:
        print("Invalid mode.")
        layer = None

    return layer


def output_image(layer: Image.Image, name, target_folder):
    if target_folder is None:
        target_folder = 'output'
        print(f"Target folder is not specified, use default folder: {target_folder}")
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Target folder does not exist, create folder: {target_folder}")

    layer.save(os.path.join(target_folder, name))
