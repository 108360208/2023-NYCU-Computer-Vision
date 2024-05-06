import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.ndimage import filters

from Harris_Corner_Detection import gaussian_smooth, sobel_edge_detection, structure_tensor, NMS, rotate

if __name__ == '__main__':
    sigma=7
    k=0.18

    img_path = os.path.join('data/chess_1.jpg')
    img = cv2.imread(img_path)
    img_copy = np.copy(img)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_filtered_K10 = convolve(img_Gray, gaussian_smooth(size=10,sigma=sigma))
    img_filtered_K10 =  img_filtered_K10 / np.amax(img_filtered_K10) * 255

    
    gradient_magnitude_K10, gradient_direction_K10 = sobel_edge_detection(img_filtered_K10)
    gradient_magnitude_K10 =  gradient_magnitude_K10 / np.amax(gradient_magnitude_K10) * 255
    gradient_direction_K10 =  gradient_direction_K10 / np.amax(gradient_direction_K10) * 255

    harrisim=structure_tensor(gradient_magnitude_K10, gradient_direction_K10, k , sigma,size=10) #harrisim=structure_tensor(gradient_magnitude_K10, gradient_direction_K10, k, sigma)
    window_size=18
    arr_flat = harrisim.flatten()
    sorted_arr = np.sort(arr_flat)
    k = 1000
    kth_largest = sorted_arr[-k]
    threshold = kth_largest
    NMS_W3=NMS(harrisim,window_size,threshold)
    plt.figure();plt.gray();plt.figure(figsize=(20,10))
    plt.imshow(img_copy)
    plt.plot([p[1] for p in NMS_W3],[p[0]for p in NMS_W3],'o')
    plt.axis('off')
    plt.savefig("output/corner_detection_results_1.jpg")
 

    sigma=10
    k=0.2

    img_path = os.path.join('data/chess_2.jpg')
    img = cv2.imread(img_path)
    img_copy = np.copy(img)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_filtered_K10 = convolve(img_Gray, gaussian_smooth(size=10,sigma=sigma))
    img_filtered_K10 =  img_filtered_K10 / np.amax(img_filtered_K10) * 255


    gradient_magnitude_K10, gradient_direction_K10 = sobel_edge_detection(img_filtered_K10)
    gradient_magnitude_K10 =  gradient_magnitude_K10 / np.amax(gradient_magnitude_K10) * 255
    gradient_direction_K10 =  gradient_direction_K10 / np.amax(gradient_direction_K10) * 255

    harrisim=structure_tensor(gradient_magnitude_K10, gradient_direction_K10, k , sigma,size=10) #harrisim=structure_tensor(gradient_magnitude_K10, gradient_direction_K10, k, sigma)
    window_size=38
    arr_flat = harrisim.flatten()
    sorted_arr = np.sort(arr_flat)
    k = 2700
    kth_largest = sorted_arr[-k]
    threshold = kth_largest
    NMS_W3=NMS(harrisim,window_size,threshold)
    plt.figure();plt.gray();plt.figure(figsize=(20,10))
    plt.imshow(img_copy)
    plt.plot([p[1] for p in NMS_W3],[p[0]for p in NMS_W3],'o')
    plt.axis('off')
    plt.savefig("output/corner_detection_results_2.jpg")
    
 