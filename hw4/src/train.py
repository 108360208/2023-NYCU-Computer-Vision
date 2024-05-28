import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from dataset import SofaDataset


with open('config.json', 'r') as f:
    config = json.load(f)

class_info = {int(cls_id): info for cls_id, info in config["classes"].items()}
class_names = [info["name"] for info in class_info.values()]
class_colors = [info["color"] for info in class_info.values()]


num_classes = 6

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


gt_transform = transforms.Compose([
    transforms.Resize((256, 256)),
])

train_dataset = SofaDataset('data/train/input', 'data/train/GT', transform, gt_transform)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

model = deeplabv3_resnet101(pretrained=True)

model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

# # Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# # Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training Loop
num_epochs = 40
model.train()
for epoch in range(num_epochs):
    for images, masks ,_ in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        if(epoch%3==0):
            torch.save(model.state_dict(), f'deep_model_{epoch}.pth')    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

