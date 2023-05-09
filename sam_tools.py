import time
import torch
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

# SAM models and their paths
models = {
    'vit_h': 'models/sam_vit_h_4b8939.pth',
    'vit_l': 'models/sam_vit_l_0b3195.pth',
    'vit_b': 'models/sam_vit_b_01ec64.pth'
}


def contour_with_points(image, point_coords, point_labels,
                        multi_mask=None, model_type=None, model_path=None, device=None):
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
    print(f"Segmentation with points completed. Time：{current_time2}，time spent：{spent_time}s.")

    return masks, scores, logits


def contour_with_box(image, box, model_type=None, model_path=None, device=None):
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
    print(f"Segmentation with box completed. Time：{current_time2}，time spent：{spent_time}s。")

    return masks, scores, logits


def auto_contour(image, model_type=None, model_path=None, device=None):
    """全自动分割"""
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
    print(f"Auto-segmentation completed. Time：{current_time2}，time spent：{spent_time}s。")
    return masks
