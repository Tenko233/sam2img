from PIL import Image
import numpy as np

from sam_tools import *
from image_tools import *

# set parameters here
input_image = "input/1.jpg"
output_folder = "output"
model = "vit_h"  # "vit_h", "vit_b" or "vit_l"
segmentation_method = "point"  # "point", "box", "auto", "all"
multi_output_mask = True  # only available for "point" or "box" mode
output_layer = "cut"  # "cut", "mask only" or "both"

input_points = [[700, 50], [2000, 1000]]  # coordinates of point prompts, only available for "point" mode
input_labels = [0, 1]  # 0 for in the ROI and 1 for out,only available for "point" mode

input_box = [100, 500, 1000, 1500]  # coordinates of the diagonal vertexes of the box, only available for "box" mode
# end of parameters

# load image and check whether the points and box are inside the image
image = Image.open(input_image)
size = image.size

if input_points:
    for point in input_points:
        if not is_in_image(point, size):
            raise Exception("Point coordinates are not in the image.")
if input_box:
    vertexes = [[input_box[0], input_box[1]], [input_box[2], input_box[3]]]
    for vertex in vertexes:
        if not is_in_image(vertex, size):
            raise Exception("Box coordinates are not in the image.")

# do segmentation task and generate layers
array = np.array(image).astype(np.uint8)
input_points = np.array(input_points)
input_labels = np.array(input_labels)
input_box = np.array(input_box)

if segmentation_method == "point" or "all":
    masks, _, _ = contour_with_points(array, input_points, input_labels, multi_output_mask, model)
    for mask in masks:
        layer, mask = mask_to_layer(image, mask, output_layer)
        psd = add_layer_and_mask(psd, layer, mask)

