import math
import cv2
import random as rd
import numpy as np

def depth_from_disparity_map(baseline, f, img):
    
    depth_map = np.zeros((img.shape[0], img.shape[1]))
    depth_array = np.zeros((img.shape[0], img.shape[1]))

    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            depth_map[i][j] = 1/img[i][j]
            depth_array[i][j] = baseline*f/img[i][j]

    return depth_map, depth_array


### compute F-matrix using 8-point algo
def get_F_matrix(list_kp1, list_kp2):
    
    A = np.zeros([len(list_kp1), 9])

    for i in range(len(list_kp1)):
        x1, y1 = list_kp1[i][0], list_kp1[i][1]
        x2, y2 = list_kp2[i][0], list_kp2[i][1]
        A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    U, s, Vt = np.linalg.svd(A)
    F = Vt[-1,:]
    F = F.reshape(3,3)
   
    # Downgrading the rank of F matrix from 3 to 2
    Uf, Df, Vft = np.linalg.svd(F)
    Df[2] = 0
    s = np.zeros((3,3))
    for i in range(3):
        s[i][i] = Df[i]

    F = np.dot(Uf, np.dot(s, Vft))
    return F



### Using RANSAC to find best F-matrix
def best_F_matrix(kp_list):

    kp_left = kp_list[0]
    kp_right = kp_list[1]
    pairs = list(zip(kp_left, kp_right))  
    
    max_inliers = 20
    threshold = 0.05  
    
    Best_F = get_F_matrix(kp_left , kp_right)
    
    
    for i in range(1000):
        pairs = rd.sample(pairs, 8)  
        rd_list_kp1, rd_list_kp2 = zip(*pairs) 
        F = get_F_matrix(rd_list_kp1, rd_list_kp2)
        tmp_inliers_img1 = []
        tmp_inliers_img2 = []

        for i in range(len(kp_left)):
            img1_x = np.array([kp_left[i][0], kp_left[i][1], 1])
            img2_x = np.array([kp_right[i][0], kp_right[i][1], 1])
            distance = abs(np.dot(img2_x.T, np.dot(F,img1_x)))
            

            if distance < threshold:
                tmp_inliers_img1.append(kp_left[i])
                tmp_inliers_img2.append(kp_right[i])

        num_of_inliers = len(tmp_inliers_img1)
        
        if num_of_inliers > max_inliers:
            max_inliers = num_of_inliers
            Best_F = F
            inliers_img1 = tmp_inliers_img1
            inliers_img2 = tmp_inliers_img2
            

    return Best_F


 # Compute the essential matrix
def get_E_matrix(F, K1, K2):

    E = np.dot(K2.T, np.dot(F, K1))
    return E



def camera_alignments(E):
    U, S, Vt = np.linalg.svd(E)

    # Ensure U and Vt are proper rotation matrices
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Construct the four possible camera matrices
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))
    t1 = U[:, 2]
    t2 = -U[:, 2]

    camera_poses = [[R1, t1], [R1, t2], [R2, t1], [R2, t2]]
    return camera_poses



def best_cameras_alignment(camera_algmts, kp_1):

    max_len = 0
    # Calculating 3D points 
    for algmt in camera_algmts:

        inliers = []        
        for point in kp_1:
            # Chirelity check
            X = np.array([point[0], point[1], 1])
            V = X - algmt[1]
            
            condition = np.dot(algmt[0][2], V)
            if condition > 0:
                inliers.append(point)    

        if len(inliers) > max_len:
            max_len = len(inliers)
            best_camera_algmt =  algmt
    
    return best_camera_algmt


def visualize_lines(img1, img2, lines, pts1, pts2):
    img1color=img1
    img2color=img2
    r, c = img1.shape[0] , img1.shape[1]
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    
    return img1color, img2color