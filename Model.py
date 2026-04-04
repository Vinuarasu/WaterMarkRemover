import os
import math
import glob
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration & Setup
# ==========================================
# This path is verified from your file listing
DATA_DIR = 'dataset/'
OUTPUT_DIR = 'predictions'

os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
IMG_SIZE = 64
TIMESTEPS = 200 

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# 2. Dataset Definition (RECURSIVE SEARCH)
# ==========================================
class WatermarkDataset(Dataset):
    def __init__(self, root_dir, is_test=False):
        self.root_dir = root_dir
        self.is_test = is_test
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Recursively find all PNG files in the dataset folder
        
        
        if is_test:
            search_pattern = os.path.join(self.root_dir, "test/*.png")
            all_pngs = glob.glob(search_pattern, recursive=True)
            # Filter for images in the 'test' directory
            self.image_paths = sorted(all_pngs)
        else:
            search_pattern = os.path.join(self.root_dir, "train/**/*.png")
            all_pngs = glob.glob(search_pattern, recursive=True)
            # Filter for images NOT in the 'test' directory (training pairs)
            self.image_paths = sorted([p for p in all_pngs if 'clean' not in p.lower()])
            self.clean_image_paths = sorted([p for p in all_pngs if 'clean' in p.lower()])
            
        print(f"Initialized {'Test' if is_test else 'Train'} Dataset: Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")

        # Handle 128x64 concatenated images (Watermarked | Clean)
        if not self.is_test:
            cimg_path = self.clean_image_paths[idx]
            clean_img = Image.open(cimg_path).convert("RGB")
            watermarked_img = img
        else:
            # Single image format (Standard for test set)
            watermarked_img = img
            clean_img = img
            
        if self.is_test:
            return self.transform(watermarked_img), filename
        else:
            return self.transform(watermarked_img), self.transform(clean_img)

# ==========================================
# 3. Model Architecture (Conditional UNet)
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x, t_emb):
        h = self.bn1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t_emb))
        h = h + time_emb[..., None, None]
        h = self.bn2(self.relu(self.conv2(h)))
        return h

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.down1 = DoubleConv(in_channels, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = DoubleConv(128, 256, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv1 = DoubleConv(256, 128, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64, time_emb_dim)
        self.out = nn.Conv2d(64, out_channels, 1)
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        mid = self.mid(self.pool2(x2), t_emb)
        u1 = self.up1(mid)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up_conv1(u1, t_emb)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up_conv2(u2, t_emb)
        return self.out(u2)

# ==========================================
# 4. Diffusion Logic
# ==========================================
beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def forward_diffusion(x_0, t, device):
    noise = torch.randn_like(x_0)
    a_bar_t = alpha_bar[t].view(-1, 1, 1, 1).to(device)
    x_t = torch.sqrt(a_bar_t) * x_0 + torch.sqrt(1 - a_bar_t) * noise
    return x_t, noise


def reverse_transform(tensor):
        tensor = (tensor + 1) / 2.0
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).byte()
        return tensor.permute(1, 2, 0).cpu().numpy()
# ==========================================
# 5. Training Engine
# ==========================================

if __name__ == '__main__':
    train_dataset = WatermarkDataset(DATA_DIR, is_test=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = ConditionalUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    history_loss = []
    print("\n--- Starting Model Training ---")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for watermarked, clean in pbar:
            watermarked, clean = watermarked.to(DEVICE), clean.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (clean.shape[0],), device=DEVICE).long()
            x_t, noise = forward_diffusion(clean, t, DEVICE)
            model_input = torch.cat([x_t, watermarked], dim=1)
            predicted_noise = model(model_input, t)
            loss = criterion(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * clean.size(0)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        history_loss.append(running_loss / len(train_dataset))

    torch.save(model.state_dict(), "Model/model.pth")
    # Save Loss Graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), history_loss, marker='o', color='crimson')
    plt.title('Diffusion Training Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss_progression.png')

    # ==========================================
    # 6. Inference & Sampling
    # ==========================================
    print("\n--- Generating Predictions ---")
    test_dataset = WatermarkDataset(DATA_DIR, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    with torch.no_grad():
        for watermarked, filenames in tqdm(test_loader, desc="Inference"):
            watermarked = watermarked.to(DEVICE)
            b = watermarked.shape[0]
            x = torch.randn((b, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
            for i in reversed(range(TIMESTEPS)):
                t = torch.full((b,), i, dtype=torch.long, device=DEVICE)
                model_input = torch.cat([x, watermarked], dim=1)
                pred_noise = model(model_input, t)
                a_t = alpha[t].view(-1, 1, 1, 1).to(DEVICE)
                a_bar_t = alpha_bar[t].view(-1, 1, 1, 1).to(DEVICE)
                b_t = beta[t].view(-1, 1, 1, 1).to(DEVICE)
                z = torch.randn_like(x) if i > 0 else 0
                x = (1 / torch.sqrt(a_t)) * (x - ((1 - a_t) / torch.sqrt(1 - a_bar_t)) * pred_noise) + torch.sqrt(b_t) * z
            for idx in range(b):
                Image.fromarray(reverse_transform(x[idx])).save(os.path.join(OUTPUT_DIR, filenames[idx]))

    # ==========================================
    # 7. Final submission.csv
    # ==========================================
    submission_data = []
    predicted_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])

    for filename in tqdm(predicted_files, desc="Encoding Results"):
        with open(os.path.join(OUTPUT_DIR, filename), "rb") as f:
            encoded_str = base64.b85encode(f.read()).decode("utf-8")
        submission_data.append({
            "datapointID": filename, 
            "subtaskID": 1, 
            "answer": encoded_str
        })

    pd.DataFrame(submission_data).to_csv('submission.csv', index=False)
    print("\nComplete! 'submission.csv' generated.")