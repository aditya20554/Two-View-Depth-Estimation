import numpy as np
import cv2

"Sum of squared distance"
def ssd(pixel_vals_1, pixel_vals_2):
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum((pixel_vals_1 - pixel_vals_2)**2)

# "Sum of absolute distance"
# def ssd(pixel_vals_1, pixel_vals_2):
#     if pixel_vals_1.shape != pixel_vals_2.shape:
#         return -1

#     return np.sum(np.absolute(pixel_vals_1 - pixel_vals_2))

"To Compare left and right windows and find min ssd value for the pixels"
def block_comparison(y, x, block_left, right_array, block_size, x_search_block_size, y_search_block_size):
    # Get search range for the right image
    x_min = max(0, x - x_search_block_size)
    x_max = min(right_array.shape[1], x + x_search_block_size)
    y_min = max(0, y - y_search_block_size)
    y_max = min(right_array.shape[0], y + y_search_block_size)
    
    first = True
    min_ssd = None
    min_index = None

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            block_right = right_array[y: y+block_size, x: x+block_size]
            ssd_ = ssd(block_left, block_right)
            if first:
                min_ssd = ssd_
                min_index = (y, x)
                first = False
            else:
                if ssd_ < min_ssd:
                    min_ssd = ssd_
                    min_index = (y, x)

    return min_index



"""Correspondence applied on the whole image to compute the disparity map"""
def ssd_correspondence(img1, img2):
    block_size = 15
    x_search_block_size = 50 
    y_search_block_size = 1
    h, w = img1.shape[0],img1.shape[1]
    disparity_map = np.zeros((h, w))

    for y in range(block_size, h-block_size):
        for x in range(block_size, w-block_size):
            block_left = img1[y:y + block_size, x:x + block_size]
            index = block_comparison(y, x, block_left, img2, block_size, x_search_block_size, y_search_block_size)
            disparity_map[y, x] = abs(index[1] - x)
    
    disparity_map_unscaled = disparity_map.copy()

    # Scaling the disparity map
    max_pixel = np.max(disparity_map)
    min_pixel = np.min(disparity_map)

    for i in range(disparity_map.shape[0]):
        for j in range(disparity_map.shape[1]):
            disparity_map[i][j] = int((disparity_map[i][j]*255)/(max_pixel-min_pixel))
    
    disparity_map_scaled = disparity_map
    return disparity_map_unscaled, disparity_map_scaled
