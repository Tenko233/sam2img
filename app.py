from sam_tools import *
from image_tools import *

# set parameters here
input_image = "input/2.jpg"
output_folder = "output"
model = "vit_h"  # "vit_h", "vit_b" or "vit_l"
segmentation_methods = ["point", "box"]  # "point", "box", "auto", "all"
multi_output_mask = True  # only available for "point" or "box" mode
output_layer = "crop"  # "crop", "mask" or "both"
img_format = "png"  # "PNG" or "JPG"， no matter it is upper or lower case

input_points = [[500, 50], [100, 1000]]  # coordinates of point prompts, only available for "point" mode
input_labels = [1, 0]  # 1 for in the ROI and 0 for out,only available for "point" mode

# coordinates of the diagonal vertexes of the box: (x1, y1, x2, y2), only available for "box" mode
input_box = [100, 500, 600, 800]
# end of parameters

# load image and check whether the points and box are inside the image
image = Image.open(input_image)
size = image.size

if input_points:
    for point in input_points:
        if not is_in_image(point, size):
            raise Exception("Point coordinates are not inside the image.")
if input_box:
    vertexes = [[input_box[0], input_box[1]], [input_box[2], input_box[3]]]
    for vertex in vertexes:
        if not is_in_image(vertex, size):
            raise Exception("Box coordinates are not inside the image.")

# pre-process the image and prompts
array = np.array(image).astype(np.uint8)
input_points = np.array(input_points)
input_labels = np.array(input_labels)
input_box = np.array(input_box)

# output the original image and prompts
plt.imshow(image)
plt.savefig(os.path.join(output_folder, "original_image." + img_format.lower()))

plt.imshow(image)
show_points(input_points, input_labels, plt.gca())
plt.savefig(os.path.join(output_folder, "point_prompt." + img_format.lower()))
plt.clf()

plt.imshow(image)
show_box(input_box, plt.gca())
plt.savefig(os.path.join(output_folder, "box_prompt." + img_format.lower()))
plt.clf()

# start segmentation
if "point" in segmentation_methods:
    masks, _, _ = seg_with_points(array, input_points, input_labels, multi_output_mask, model)
    for i in range(len(masks)):
        if output_layer == "crop" or "both":
            layer = mask_to_layer(image, masks[i], "crop")
            name = "point_crop_" + str(i) + "." + img_format.lower()
            output_image(layer, name, output_folder)
        if output_layer == "mask" or "both":
            layer = mask_to_layer(image, masks[i], "mask")
            name = "point_mask_" + str(i) + "." + img_format.lower()
            output_image(layer, name, output_folder)

if "box" in segmentation_methods:
    masks, _, _ = seg_with_box(array, input_box, model)
    for i in range(len(masks)):
        if output_layer == "crop" or "both":
            layer = mask_to_layer(image, masks[i], "crop")
            name = "box_crop_" + str(i) + "." + img_format.lower()
            output_image(layer, name, output_folder)
        if output_layer == "mask" or "both":
            layer = mask_to_layer(image, masks[i], "mask")
            name = "box_mask_" + str(i) + "." + img_format.lower()
            output_image(layer, name, output_folder)

if "auto" in segmentation_methods:
    masks = auto_seg(array, model)
    for i in range(len(masks)):
        if output_layer == "crop" or "both":
            layer = mask_to_layer(image, masks[i]['segmentation'], "crop")
            name = "auto_crop_" + str(i) + "." + img_format.lower()
            output_image(layer, name, output_folder)
        if output_layer == "mask" or "both":
            layer = mask_to_layer(image, masks[i]['segmentation'], "mask")
            name = "auto_mask_" + str(i) + "." + img_format.lower()
            output_image(layer, name, output_folder)

print(f"Done! Output images are saved in {output_folder}.")
