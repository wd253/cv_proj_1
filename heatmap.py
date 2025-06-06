import os
import json
import torch
import numpy as np
import cv2
import math
from tqdm import tqdm

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    return max(0, int(min(r1, 100)))

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = cv2.getGaussianKernel(diameter, sigma=diameter / 6)
    gaussian = np.outer(gaussian, gaussian)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if masked_gaussian.shape == masked_heatmap.shape:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def convert_labelme_to_centernet(data, down=4, num_classes=4, max_objs=128):
    W, H = data["imageWidth"], data["imageHeight"]
    output_w, output_h = W // down, H // down

    heatmap = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    reg = np.zeros((max_objs, 2), dtype=np.float32)
    wh = np.zeros((max_objs, 2), dtype=np.float32)
    ind = np.zeros((max_objs), dtype=np.int64)
    reg_mask = np.zeros((max_objs), dtype=np.uint8)

    label2id = {
        "standing": 0,
        "sitting": 1,
        "lying": 2,
        "throwing": 3
    }

    num_obj = 0
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in label2id:
            continue
        cls_id = label2id[label]
        pt1, pt2 = shape["points"]
        x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
        x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        cx_out, cy_out = cx / down, cy / down
        int_cx, int_cy = int(cx_out), int(cy_out)

        radius = gaussian_radius((math.ceil(h / down), math.ceil(w / down)))
        draw_umich_gaussian(heatmap[cls_id], [int_cx, int_cy], radius)

        if num_obj >= max_objs:
            continue
        wh[num_obj] = [w / down, h / down]
        reg[num_obj] = [cx_out - int_cx, cy_out - int_cy]
        ind[num_obj] = int_cy * output_w + int_cx
        reg_mask[num_obj] = 1
        num_obj += 1

    return {
        "heatmap": torch.tensor(heatmap),
        "reg": torch.tensor(reg),
        "wh": torch.tensor(wh),
        "ind": torch.tensor(ind),
        "reg_mask": torch.tensor(reg_mask)
    }

def process_all_jsons(label_dir, output_dir, down=4):
    os.makedirs(output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]

    for json_file in tqdm(json_files, desc="Converting JSON to Heatmap"):
        json_path = os.path.join(label_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        targets = convert_labelme_to_centernet(data, down=down)
        base_name = os.path.splitext(json_file)[0]
        torch.save(targets, os.path.join(output_dir, f"{base_name}.pt"))
