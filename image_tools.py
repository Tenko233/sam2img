import os
import base64
import io
import random
import shelve
import numpy as np
from PIL import Image, ImageDraw


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


def export_image_to_file(layer: Image.Image, name, target_folder):
    if target_folder is None:
        target_folder = 'output'
        print(f"Target folder is not specified, use default folder: {target_folder}")
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Target folder does not exist, create folder: {target_folder}")

    layer.save(os.path.join(target_folder, name))


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    image = image.convert('RGBA')

    return image, filename


def draw_points(image, points, labels, size=1):
    img_size = image.size
    relative_size = size * np.max(img_size) / 100
    print(relative_size)
    half_size = round(relative_size / 2)
    pos_points = points[labels == 1]
    neg_points = points[labels == 0]
    draw = ImageDraw.Draw(image)
    for p in pos_points:
        draw.ellipse([p[0] - half_size, p[1] - half_size, p[0] + half_size, p[1] + half_size],
                     fill=(0, 255, 0, 255),
                     outline=(0, 0, 0, 255))
    for p in neg_points:
        draw.ellipse([p[0] - half_size, p[1] - half_size, p[0] + half_size, p[1] + half_size],
                     fill=(255, 0, 0, 255),
                     outline=(0, 0, 0, 255))

    return image


def draw_box(image, box, width=1):
    img_size = image.size
    relative_width = round(width * np.max(img_size) / 100)
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=(0, 0, 255, 255), width=relative_width)

    return image


def draw_mask(image, masks, rgba=None):
    if rgba is None:
        random_color = True
    else:
        random_color = False

    for mask in masks:
        if random_color:
            rgba = [random.randint(0, 255) for _ in range(3)]
            rgba.append(192)

        mask = np.array(mask)
        mask_image = np.stack(
            (mask * rgba[0], mask * rgba[1], mask * rgba[2], mask * rgba[3]),
            axis=2
        )
        foreground = Image.fromarray(mask_image.astype(np.uint8))
        foreground.convert('RGBA')
        image.paste(foreground, (0, 0), foreground)

    return image


def save_temp_data(data, key):
    temp = '.temp/'
    if not os.path.exists(temp):
        os.makedirs(temp)
    with shelve.open(temp+'.temp') as db:
        db[key] = data


def load_temp_data(key):
    temp = '.temp/'
    with shelve.open(temp+'.temp') as db:
        data = db[key]
        return data
