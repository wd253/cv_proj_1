import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from model.centernet import CenterNet
from tqdm import tqdm

class CenterNetDataset(Dataset):
    def __init__(self, image_dir, heatmap_dir):
        self.image_dir = image_dir
        self.heatmap_dir = heatmap_dir
        self.filenames = [f.replace('.pt', '') for f in os.listdir(heatmap_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image_path = os.path.join(self.image_dir, fname + '.png')
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.resize(image, (512, 512)).astype(np.float32) / 255.0
        # 정규화 (ImageNet 기준)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = torch.from_numpy(image).permute(2, 0, 1)

        heatmap = torch.load(os.path.join(self.heatmap_dir, fname + '.pt'))  # (C, H, W)
        if heatmap.shape[-1] != 128:
            heatmap = nn.functional.interpolate(
                heatmap.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
            ).squeeze(0)

        return image.float(), heatmap.float()


def focal_loss(preds, targets, alpha=2, beta=4):
    preds = torch.clamp(torch.sigmoid(preds), 1e-4, 1 - 1e-4)  # 안정화
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, beta)

    pos_loss = torch.log(preds) * torch.pow(1 - preds, alpha) * pos_inds
    neg_loss = torch.log(1 - preds) * torch.pow(preds, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum().float()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CenterNet(num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = CenterNetDataset('CV_train/Images', 'CV_train/Heatmaps')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    model.train()
    for epoch in range(30):
        total_loss = 0
        for imgs, heatmaps in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)
            preds, _, _ = model(imgs)
            loss = focal_loss(preds, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f'train/centernet_epoch{epoch+1}.pth')


if __name__ == '__main__':
    train()
