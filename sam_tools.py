import time
import matplotlib.pyplot as plt
import torch
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

# SAM models and their paths
models = {
    'vit_h': 'models/sam_vit_h_4b8939.pth',
    'vit_l': 'models/sam_vit_l_0b3195.pth',
    'vit_b': 'models/sam_vit_b_01ec64.pth'
}


def check_cuda():
    return torch.cuda.is_available()


def seg_with_points(image, point_coords, point_labels,
                    multi_mask=None, model_type=None, model_path=None, device=None,
                    need_info=False):
    if multi_mask is None:
        multi_mask = True
    if model_type is None:
        model_type = "vit_h"
    if model_path is None:
        model_path = models[model_type]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # use GPU as default

    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    current_time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time1 = time.time()
    print(f"Point-prompt model loaded.Time：{current_time1}，device：{device}.")

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multi_mask,
    )
    current_time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time2 = time.time()
    spent_time = round(time2 - time1, 3)
    info = f"Segmentation with points completed. Time：{current_time2}，time spent：{spent_time}s."
    print(info)
    torch.cuda.empty_cache()

    if need_info:
        return masks, scores, logits, info
    else:
        return masks, scores, logits


def seg_with_box(image, box, model_type=None, model_path=None, device=None, need_info=False):
    if model_type is None:
        model_type = "vit_h"
    if model_path is None:
        model_path = models[model_type]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    current_time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time1 = time.time()
    print(f"Box-prompt model loaded. Time：{current_time1}，device：{device}.")

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=True,
    )
    current_time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time2 = time.time()
    spent_time = round(time2 - time1, 3)
    info = f"Segmentation with box completed. Time：{current_time2}，time spent：{spent_time}s."
    print(info)
    torch.cuda.empty_cache()

    if need_info:
        return masks, scores, logits, info
    else:
        return masks, scores, logits


def auto_seg(image, model_type=None, model_path=None, device=None, need_info=False):
    if model_type is None:
        model_type = "vit_h"
    if model_path is None:
        model_path = models[model_type]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    current_time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time1 = time.time()
    print(f"Auto-segmentation model loaded. Time：{current_time1}，device：{device}。")

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    current_time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time2 = time.time()
    spent_time = round(time2 - time1, 3)
    info = f"Auto-segmentation completed. Time：{current_time2}，time spent：{spent_time}s。"
    print(info)
    torch.cuda.empty_cache()

    if need_info:
        return masks, info
    else:
        return masks


# The following two functions are copied from SAM's demo
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
