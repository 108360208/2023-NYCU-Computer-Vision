import matplotlib.pyplot as plt
import numpy as np
import cv2
# %matplotlib inline
from scipy.ndimage import convolve

from scipy import ndimage

def gaussian_smooth(size, sigma=1):
    ########################################################################
    # TODO:                                                                #
    #   Perform the Gaussian Smoothing                                     #
    #   Input: window size, sigma                                          #
    #   Output: smoothing image                                            #
    ########################################################################
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return kernel

# from scipy.ndimage.filters import convolve
# img_filtered_K5 = convolve(img_Gray, gaussian_smooth(size=5,sigma=5))
# img_filtered_K10 = convolve(img_Gray, gaussian_smooth(size=10,sigma=5))

def sobel_edge_detection(im):
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    
    Ix = convolve(im, sobel_x)
    Iy = convolve(im, sobel_y)

    
    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)
    gradient_direction = np.arctan2(Iy, Ix) * (180 / np.pi)  # 將弧度轉換為度
    return  (gradient_magnitude, gradient_direction)

def structure_tensor(gradient_magnitude, gradient_direction, k, sigma,size):
    Ix = gradient_magnitude * np.cos(np.deg2rad(gradient_direction))
    Iy = gradient_magnitude * np.sin(np.deg2rad(gradient_direction))

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    g = gaussian_smooth(size, sigma)  

    Sxx = convolve(Ixx, g)
    Syy = convolve(Iyy, g)
    Sxy = convolve(Ixy, g)
    
    detM = Sxx * Syy - Sxy**2
    traceM = Sxx + Syy

    StructureTensor = detM - k * (traceM**2)

    return  StructureTensor

import numpy as np


def maximum_filter(image, size):
    rows, cols = image.shape

    filtered_image = np.zeros_like(image)
    
    r = size // 2
    
    for i in range(rows):
        for j in range(cols):

            start_i = max(0, i - r)
            end_i = min(rows, i + r + 1)
            start_j = max(0, j - r)
            end_j = min(cols, j + r + 1)
            region = image[start_i:end_i, start_j:end_j]

            filtered_image[i, j] = np.max(region)
    
    return filtered_image

def NMS(harrisim, window_size=30, threshold=0.1):

    mask = np.zeros_like(harrisim, dtype=bool)

    r = window_size // 2

    local_max = maximum_filter(harrisim, size=window_size) == harrisim

    thresholded_max = (harrisim >= threshold)
    
    candidate_coords = np.logical_and(local_max, thresholded_max)
    
    mask[candidate_coords] = True
    
    filtered_coords = np.argwhere(mask)
    
    return filtered_coords



def plot_harris_points(image,filtered_coords):
    plt.figure()
    plt.gray()
    plt.figure(figsize=(20,10))
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0]for p in filtered_coords],'+')
    plt.axis('off')
    plt.show()
    
def rotate(image, angle, center = None, scale = 1.0):

    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated