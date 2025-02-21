#assignment1.py: driver for hybrid images project.

import cv2
import sys
from alignment import aligner_affine, aligner_homography, read_keypoints
from frequency_fusion import frequency_hybrid_images
from spatial_fusion import spatial_hybrid
import numpy as np

#driver 
def main():
    #read images and convert it to grayscale.
    img1 = cv2.imread('image1.jpg') #DIMENSIONS: height is image1.shape[0], width is image1.shape[1]
    img2 = cv2.imread('image2.jpg')
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #read keypoints
    if len(sys.argv) != 3:
        print("Usage: python assignment1.py keypoints1 keypoints2")
        sys.exit(1)
    kp1 = read_keypoints(sys.argv[1])
    kp2 = read_keypoints(sys.argv[2])

    #alignment of image 2.
    aligned_img = aligner_affine(img1_gray, img2_gray, kp1, kp2)
 
    #frequency fusion
    result = frequency_hybrid_images(img1_gray, aligned_img)

    #spatial fusion
    result2 = spatial_hybrid(img1_gray, aligned_img)
    
main()

#useful for testing.
#cv2.imshow("Aligned Image", aligned_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
