#alignment.py
import cv2
import numpy as np

#https://www.geeksforgeeks.org/image-registration-using-opencv-python/
def aligner_homography(image1, image2, kp1, kp2):
    #check if keypoints are provided or of same size (must or unaligned)
    if not kp1.size or not kp2.size or kp1.shape != kp2.shape:
        print("No keypoints provided. Resizing image2 to match image1 dimensions.")
        aligned = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        cv2.imwrite('aligned.jpg', aligned) 
        return aligned
    
    num_points = min(kp1.shape[0], kp2.shape[0]) #they are both the same size so this is redundant.
    p1 = kp1[:num_points].astype(np.float32)
    p2 = kp2[:num_points].astype(np.float32)
    
    H, mask = cv2.findHomography(p2, p1, cv2.RANSAC)     #get homography that maps points from image2 (p2) to image1 (p1)
    
    #warp image2 using the homography matrix so it aligns with image1.
    height, width = image1.shape[:2]
    aligned = cv2.warpPerspective(image2, H, (width, height))
    cv2.imwrite('aligned.jpg', aligned)
    return aligned

#https://www.geeksforgeeks.org/python-opencv-affine-transformation/?ref=header_outind
def aligner_affine(image1, image2, kp1, kp2):
    #check for keypoints; if missing, resize image2 to image1's size.
    if not kp1.size or not kp2.size or kp1.shape != kp2.shape:
        print("Not enough keypoints. Resizing image2 to match image1.")
        aligned = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        cv2.imwrite('aligned.jpg', aligned)
        return aligned
    #use min number of keypoints if the arrays differ in size.
    num_points = min(kp1.shape[0], kp2.shape[0])
    p1 = kp1[:num_points]  # Destination keypoints from image1.
    p2 = kp2[:num_points]  # Source keypoints from image2.
    #est affine transformation from source (p2) to destination (p1) keypoints.
    M, inliers = cv2.estimateAffinePartial2D(p2, p1)
    #apply transformation to image2.
    height, width = image1.shape[:2]
    aligned = cv2.warpAffine(image2, M, (width, height))
    cv2.imwrite('aligned.jpg', aligned)
    return aligned

