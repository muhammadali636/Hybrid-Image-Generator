#alignment.py: here we have affine alignment and homography (nonrigid) functions. I did both to learn and test for my images. In future try thin plate splines. 

import cv2
import numpy as np


#read keypoint files. 
def read_keypoints(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return np.array([], dtype=np.float32)

    #if empty file return an empty arr.
    if not lines:
        return np.array([], dtype=np.float32)
    kp= [] #keypoints array

    #parser return the keypoint array.
    for line in lines:
        line = line.strip()
        #skip empty lines
        if not line:
            continue  
        #remove brackets
        if line.startswith('(') and line.endswith(')'):
            line = line[1:-1]      
        parts = line.split(',')
        #read x, y coords and append to kp array
        x = float(parts[0].strip())
        y = float(parts[1].strip())
        kp.append((x, y))

    return np.array(kp, dtype=np.float32)

#affine (rigid) alignment - referenced: https://www.geeksforgeeks.org/python-opencv-affine-transformation/?ref=header_outind
def aligner_affine(image1, image2, kp1, kp2):
    #check for keypoints; if missing or not equal, resize image2 to image1's size.
    if not kp1.size or not kp2.size or kp1.shape != kp2.shape:
        print("Not enough keypoints. Resizing image2 to match image1.")
        aligned = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        cv2.imwrite('output_images/aligned.jpg', aligned)
        return aligned
    num_points = min(kp1.shape[0], kp2.shape[0])  #use min number of keypoints if the arrays differ in size. They are the same so this is redundant.
    p1 = kp1[:num_points]  #keypoints from image1.
    p2 = kp2[:num_points]  #keypoints from image2.
    #est affine transformation from source (p2) to destination (p1) keypoints.
    M, inliers = cv2.estimateAffinePartial2D(p2, p1)
    #apply transformation to image2.
    height, width = image1.shape[:2]
    aligned = cv2.warpAffine(image2, M, (width, height))
    cv2.imwrite('output_images/aligned.jpg', aligned)
    return aligned

#homography alignment (nonrigid)  - referenced: https://www.geeksforgeeks.org/image-registration-using-opencv-python/. Doesnt work well, TODO: thin plate splines.
def aligner_homography(image1, image2, kp1, kp2):
    #check if keypoints are provided or of same size (must or unaligned)
    if not kp1.size or not kp2.size or kp1.shape != kp2.shape:
        print("No keypoints provided. Resizing image2 to match image1 dimensions.")
        aligned = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        cv2.imwrite('output_images/aligned.jpg', aligned) 
        return aligned

    #compute homography matrix
    num_points = min(kp1.shape[0], kp2.shape[0]) #they are both the same size so this is redundant.
    p1 = kp1[:num_points].astype(np.float32)
    p2 = kp2[:num_points].astype(np.float32)
    H, mask = cv2.findHomography(p2, p1, cv2.RANSAC)    #get homography that maps points from image2 (p2) to image1 (p1)
    
    #warp image2 using homography matrix to align with image1
    height, width = image1.shape[:2]
    aligned = cv2.warpPerspective(image2, H, (width, height))
    cv2.imwrite('output_images/aligned.jpg', aligned) #aligned img2.
    return aligned
