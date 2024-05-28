import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image

num_classes = 6

class SofaDataset(Dataset):
    def __init__(self, image_dir, GT_dir, transform=None, gt_transform=None):
        self.image_dir = image_dir
        self.GT_dir = GT_dir
        self.transform = transform
        self.gt_transform = gt_transform
        self.images = os.listdir(image_dir)
       
    def __len__(self):
        return len(self.images)
   
    def __getitem__(self, idx):
        img_gt = self.images[idx].split('.')[0] + "_pix." + self.images[idx].split('.')[1]
        img_path = os.path.join(self.image_dir, self.images[idx])
        GT_path = os.path.join(self.GT_dir, img_gt)
        image = Image.open(img_path).convert("RGB")
        GT = Image.open(GT_path).convert("RGB")
        if self.gt_transform:
            GT = self.gt_transform(GT)
        GT_array = np.array(GT)
        width,height,ch = GT_array.shape
        new_array = np.zeros((height, width))
        
        for i in range (height):
            for j in range(width):
                if(GT_array[j][i][0]==60):
                    new_array[i][j]=0
                elif(GT_array[j][i][0]==110):
                    new_array[i][j]=1
                elif(GT_array[j][i][0]==50):
                    new_array[i][j]=2
                elif(GT_array[j][i][0]==180):
                    new_array[i][j]=3
                elif(GT_array[j][i][0]==100):
                    new_array[i][j]=4
                else:
                    new_array[i][j]=5
                    
        one_hot_encoding = np.zeros((num_classes, height, width))

        for class_index in range(num_classes):
            one_hot_encoding[class_index][new_array == class_index] = 1
 
       
        if self.transform:
            image = self.transform(image)
           
        return image, one_hot_encoding, self.images[idx]  