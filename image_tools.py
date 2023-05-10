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
    if mode == "cut":
        array = np.array(image)
        print(array.shape)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
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
