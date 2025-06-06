import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO  # YOLOv8
from model.centernet import CenterNet
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor

# Config
IMG_DIR = 'test1/Images'
OUTPUT_DIR = 'test1/outputs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF_THRESH = 0.3
NMS_THRESH = 0.4
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models
yolo = YOLO('yolov8n.pt')
centernet = CenterNet(num_classes=4).to(DEVICE)
centernet.load_state_dict(torch.load('train/centernet_epoch30.pth', map_location=DEVICE))
centernet.eval()

# Class labels
CLASSES = ['standing', 'sitting', 'lying', 'throwing']

# Helper: run CenterNet on given image
def run_centernet(image):
    img_resized = cv2.resize(image, (512, 512))
    img_tensor = to_tensor(img_resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        heatmap, wh, offset = centernet(img_tensor)
        heatmap = heatmap.sigmoid_()
    return heatmap[0].cpu().numpy()

# Helper: decode heatmap → detections (top-K)
def decode_heatmap(heatmap, thresh=0.3, top_k=40):
    detections = []
    for cls_id in range(heatmap.shape[0]):
        cls_map = heatmap[cls_id]
        ys, xs = np.where(cls_map > thresh)
        for x, y in zip(xs, ys):
            score = cls_map[y, x]
            detections.append([x * 4, y * 4, 32, 32, score, cls_id])  # scale up
    if len(detections) == 0:
        return []
    detections = torch.tensor(detections)
    boxes = torch.cat([detections[:, :2] - detections[:, 2:4] / 2, detections[:, :2] + detections[:, 2:4] / 2], dim=1)
    scores = detections[:, 4]
    labels = detections[:, 5]
    keep = nms(boxes, scores, NMS_THRESH)
    return boxes[keep], scores[keep], labels[keep].long()

# Inference loop
for fname in os.listdir(IMG_DIR):
    if not fname.endswith(('.png', '.jpg')):
        continue
    image = cv2.imread(os.path.join(IMG_DIR, fname))

    # Step 1: YOLO coarse detection
    yolo_results = yolo.predict(image, conf=CONF_THRESH, classes=[0])  # person only
    if len(yolo_results[0].boxes) == 0:
        continue

    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    yolo_count = len(yolo_boxes)

    # Step 2: CenterNet detection
    heatmap = run_centernet(image)
    boxes, scores, labels = decode_heatmap(heatmap, thresh=CONF_THRESH)
    centernet_count = boxes.shape[0]

    # Step 3: Bounding box count matching (truncate CenterNet boxes if needed)
    if centernet_count > yolo_count:
        topk = scores.topk(yolo_count).indices
        boxes = boxes[topk]
        labels = labels[topk]

    # Step 4: Draw results
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].int().tolist()
        cls_name = CLASSES[labels[i]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), image)

print("[✔] Inference complete. Results saved to outputs/")
