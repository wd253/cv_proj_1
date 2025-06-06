import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Config
IMAGE_DIR = 'CV_train/Images'
HEATMAP_DIR = 'CV_train/Heatmaps'
OUTPUT_DIR = 'CV_train/HeatmapVisuals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['standing', 'sitting', 'lying', 'throwing']

# 시각화 함수
def visualize_heatmap(image_path, heatmap_path, class_names=None, save_path=None):
    # 공백 제거
    image_path = image_path.strip()
    heatmap_tensor = torch.load(heatmap_path)

    # 예외 처리
    if isinstance(heatmap_tensor, dict):
        heatmap_tensor = heatmap_tensor['heatmap']
    if not isinstance(heatmap_tensor, torch.Tensor):
        raise TypeError("Loaded heatmap is not a torch.Tensor")

    if heatmap_tensor.ndim != 3:
        raise ValueError(f"Expected 3D heatmap, got shape {heatmap_tensor.shape}")

    heatmap = heatmap_tensor.numpy()  # (C, H, W)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    _, output_h, output_w = heatmap.shape

    # Resize heatmap to image size
    heatmap_resized = np.zeros((heatmap.shape[0], h, w), dtype=np.float32)
    for i in range(heatmap.shape[0]):
        heatmap_resized[i] = cv2.resize(heatmap[i], (w, h))

    # Create overlay image
    fig, axes = plt.subplots(1, heatmap.shape[0] + 1, figsize=(16, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i in range(heatmap.shape[0]):
        axes[i+1].imshow(image, alpha=0.7)
        axes[i+1].imshow(heatmap_resized[i], cmap='hot', alpha=0.5)
        title = f"Class {i}" if class_names is None else class_names[i]
        axes[i+1].set_title(f"Heatmap: {title}")
        axes[i+1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 일괄 처리
if __name__ == '__main__':
    for fname in os.listdir(HEATMAP_DIR):
        if not fname.endswith('.pt'):
            continue
        heatmap_path = os.path.join(HEATMAP_DIR, fname)
        image_path = os.path.join(IMAGE_DIR, fname.replace('.pt', '.png'))

        save_name = fname.replace('.pt', '_vis.png')
        save_path = os.path.join(OUTPUT_DIR, save_name)

        try:
            visualize_heatmap(image_path, heatmap_path, class_names=CLASS_NAMES, save_path=save_path)
        except Exception as e:
            print(f"[!] Failed to process {fname}: {e}")

    print(f'[✔] Heatmap visualization complete. Check: {OUTPUT_DIR}')
