import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights, deeplabv3_resnet101
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



model = deeplabv3_resnet101(pretrained=True)
model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


test_dataset = SofaDataset('data/test/input', 'data/test/GT', transform, gt_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    for images, _, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)['out'].to(device)
        outputs=outputs.squeeze(0)
        predicted=outputs.argmax(dim=0, keepdim=True).to(device) 
        predicted = predicted.squeeze(0).cpu().numpy()
        new_array = np.zeros((3,256, 256))
        for i in range (0,256):
            for j in range (0,256):
                class_index = predicted[i][j]

                color = class_colors[(class_index)]
                new_array[0][i][j] = color[0]
                new_array[1][i][j] = color[1]
                new_array[2][i][j] = color[2]
    
        img_array = new_array.transpose(2, 1, 0)
        img_array = img_array.astype(np.uint8)  
        pred_image = Image.fromarray(img_array)
        #calculate PSNR to pred and target
        pred_image.save(f"data/test/predict/{filenames[0]}")

def calculate_iou(pred, target, num_classes = 6):
 
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target[cls] == 1
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))  
            print(f"({class_names[cls]}): IoU = NaN")
        else:
            iou = intersection / union
            ious.append(iou)
            print(f"({class_names[cls]}): IoU = {iou:.4f}")
    return ious

def mean_iou(pred, target, num_classes = 6):

    ious = calculate_iou(pred, target, num_classes)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]  
    mean_iou = np.mean(valid_ious)
    return mean_iou

# Example IoU Calculation
iou_scores = []
for images, masks, filenames in test_loader:
    images = images.to(device)
    masks = masks.to(device).long().squeeze(1)
    outputs = model(images)['out']
    predicted = outputs.argmax(dim=1).squeeze(0).cpu().numpy()
    target = Image.open(f"data/test/GT/{filenames[0].split('.')[0]}_pix.{filenames[0].split('.')[1]}").convert("RGB")
    target = target.resize((256, 256))
    pred_image = Image.open(f"data/test/predict/{filenames[0].split('.')[0]}.{filenames[0].split('.')[1]}").convert("RGB")
    mse = np.mean((np.array(target) - np.array(pred_image)) ** 2)
    if mse == 0:
        print(f"({filenames[0]}): PSNR = 100.00")
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        print(f"({filenames[0]}): PSNR = {psnr:.2f}")

    iou = mean_iou(predicted, masks.squeeze(0).cpu().numpy())
    iou_scores.append(iou)
  
print(f"Mean IoU: {(np.mean(iou_scores)):.4f}")