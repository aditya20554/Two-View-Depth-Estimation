import numpy as np
from PIL import Image
import cv2
import math
import os
import matplotlib.pyplot as plt

### Reshaping ground truth disparity file
dir= "data"

### for barplot
X = []
Y = []

for image_dir in os.listdir(dir):
    print(image_dir)
    img = cv2.imread(dir + "/" + image_dir +'/disp0.png', cv2.IMREAD_GRAYSCALE)
    
    img1 = cv2.imread(dir + "/" + image_dir + "/disparity_map_grayscale.png")
    new_height, new_width = img1.shape[0], img1.shape[1]
    

    img_reshaped = cv2.resize(img, (new_width, new_height))
    cv2.imwrite(dir + "/" + image_dir +'/gt.png', img_reshaped)

    gt = Image.open(dir + "/" + image_dir +"/gt.png").convert("L")
    generated = Image.open(dir + "/" + image_dir +"/disparity_map_grayscale.png").convert("L")

    gt_arr = np.asarray(gt)
    generated_arr = np.asarray(generated)

    # Compute the RMSE between the two arrays
    rmse = math.sqrt(np.mean((gt_arr - generated_arr) ** 2))

    # Print the RMSE
    print("Root Mean Squared Error:", rmse)
    X.append(image_dir)
    Y.append(rmse)
    print()

plt.bar(X, Y)
plt.xlabel("Image")
plt.ylabel("RMSE")
plt.show()

