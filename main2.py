#### Imports
import math
import cv2
import random as rd
import numpy as np  
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import os

from calibration import *
from rectification import *
from correspondence import *

import warnings
warnings.filterwarnings("ignore")


## Camera Parameters
K1 = np.array([[5299.313,  0,   1263.818], 
                [0,      5299.313, 977.763],
                [0,          0,       1   ]])
K2 = np.array([[5299.313,   0,    1438.004],
            [0,      5299.313,  977.763 ],
            [0,           0,      1     ]])


dir= "data"

for image_dir in os.listdir(dir):
  # if(image_dir=="artroom1" or image_dir=="artroom2" or image_dir=="bandsaw1" or image_dir=="curule1"
  #    or image_dir=="ladder2" or image_dir=="octogons1" or image_dir=="octogons2" or image_dir=="pendulum1" or image_dir=="pendulum2" or image_dir=="skates2" or image_dir=="skiboots1"):
  #   continue
  
  print(image_dir)
  img1 = cv2.imread(dir + "/" + image_dir + "/im0.png")
  img2 = cv2.imread(dir + "/" + image_dir + "/im1.png")
  
  width = int(img1.shape[1]* 0.3) 
  height = int(img1.shape[0]* 0.3)

  img1 = cv2.resize(img1, (width, height), interpolation = cv2.INTER_AREA)
  img2 = cv2.resize(img2, (width, height), interpolation = cv2.INTER_AREA)


  ## keypoints detection and feature matching
  orb = cv2.ORB_create(nfeatures=10000)
  
  # find the keypoints and descriptors with ORB
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)

  # Create a BruteForce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)

  # Match the descriptors using BruteForce
  bf_matches = bf.match(des1, des2)

  # Sort the matches by their distance
  bf_matches = sorted(bf_matches, key = lambda x:x.distance)
  bf_matches =bf_matches[:30]

  # Draw the matches
  img_with_keypoints = cv2.drawMatches(img1, kp1, img2, kp2, bf_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  # Getting x,y coordinates of the matches
  kp_1 = [list(kp1[mat.queryIdx].pt) for mat in bf_matches] 
  kp_2 = [list(kp2[mat.trainIdx].pt) for mat in bf_matches]


  # print(kp_1)
  # print(kp_2)
  

  ### Calculating fundamental matrix
  F = best_F_matrix([kp_1,kp_2])
  print("F matrix", F)
  print()
  

  ### Calculating essential matrix
  E = get_E_matrix(F, K1, K2)
  print("E matrix", E)
  print()

  
  ## getting camera alignments for E
  camera_algmts = camera_alignments(E)

  ## getting best camera alignment with max no. of inliers
  best_alignment = best_cameras_alignment(camera_algmts, kp_1)
  
  print("Rotation", best_alignment[0])
  print()
  print("Translation", best_alignment[1])
  print()
  
  pts1 = np.int32(kp_1)
  pts2 = np.int32(kp_2)

  ## rectification
  rectified_pts1, rectified_pts2, img1_rectified, img2_rectified = rectify(img1, img2, pts1, pts2, F)
  cv2.imwrite(dir+"/"+image_dir+"/rectified_1.png", img1_rectified)
  cv2.imwrite(dir+"/"+image_dir+"/rectified_2.png", img2_rectified)



  ### visualizing feature points and lines
  epilines1 = cv2.computeCorrespondEpilines(rectified_pts2.reshape(-1, 1, 2), 2, F)
  epilines1 = epilines1.reshape(-1, 3)
  featured_image1, featured_image2 = visualize_lines(img1_rectified, img2_rectified, epilines1, rectified_pts1,     rectified_pts2)

  lines2 = cv2.computeCorrespondEpilines(rectified_pts1.reshape(-1, 1, 2), 1, F)
  lines2 = lines2.reshape(-1, 3)
  featured_image3, featured_image4 = visualize_lines(img2_rectified, img1_rectified, lines2, rectified_pts2, rectified_pts1)

  cv2.imwrite(dir + "/" + image_dir + "/left_image.png", featured_image1)
  cv2.imwrite(dir + "/" + image_dir + "/right_image.png", featured_image3)


  ### Disparity map using correspondences

  disparity_map_unscaled, disparity_map_scaled = ssd_correspondence(img1_rectified, img2_rectified)
  
  plt.figure(1)
  plt.title('Disparity Map Graysacle')
  plt.imshow(disparity_map_scaled, cmap='gray')
  plt.savefig(dir+"/"+image_dir+"/disparity_map_grayscale.png")


  plt.figure(2)
  plt.title('Disparity Map Hot')
  plt.imshow(disparity_map_scaled, cmap='hot')
  plt.savefig(dir+"/"+image_dir+"/disparity_map_hot.png")


  ### Depth map from disparity map
  baseline, f = 177.288, 5299.313

  depth_map, depth_array = depth_from_disparity_map(baseline, f, disparity_map_unscaled)

  plt.figure(3)
  plt.title('Depth Map Graysacle')
  plt.imshow(depth_map, cmap='gray')
  plt.savefig(dir+"/"+image_dir+"/depth_map_grayscale.png")


  plt.figure(4)
  plt.title('Depth Map Hot')
  plt.imshow(depth_map, cmap='hot')
  plt.savefig(dir+"/"+image_dir+"/depth_map_hot.png")
  
  # plt.show()

  print("=="*20)