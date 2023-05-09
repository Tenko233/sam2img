from sam_tools import *
from image_tools import *

# Set parameters here
input_image = "input/1.jpg"
output_folder = "output"
model = "vit_h"  # "vit_h", "vit_b" or "vit_l"
segmentation_method = "point"  # "point", "box" or "auto"
multi_output_mask = True  # only available for "point" or "box" mode
output_layer = "cut"  # "cut", "mask only" or "both"
