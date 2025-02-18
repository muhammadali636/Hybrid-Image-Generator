# assignment1.py: driver.
# Spatial-Domain Blending (using Gaussian/Laplacian pyramids)
# Frequency-Domain Blending (using FFT filtering)

import cv2
from alignment import aligner_affine, aligner_homography
from frequency_fusion import frequency_hybrid_images
from spatial_fusion import spatial_hybrid
import numpy as np

def read_keypoints(filename):
    with open(filename, 'r') as file:
        content = file.read().strip()
    
    #if file is empty, return an empty array
    if not content:
        return np.array([], dtype=np.float32)
    
    #remove the outer square brackets if present
    if content[0] == '[' and content[-1] == ']':
        content = content[1:-1]
    #split the content by the delimiter between keypoints.
    kp_strings = content.split("),")
    
    kp_list = []
    for kp_str in kp_strings:
        #remove any remaining parentheses and whitespace
        kp_str = kp_str.replace("(", "").replace(")", "").strip()
        if kp_str:
            parts = kp_str.split(',')
            #expecting two parts for x and y
            x = float(parts[0].strip())
            y = float(parts[1].strip())
            kp_list.append((x, y))
    return np.array(kp_list, dtype=np.float32)

def main():

    img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE) #DIMENSIONS: print('height is', image1.shape[0]), print('image width is',image1.shape[1])
    img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("can't read {img1} or {img2}")
        return 

    kp1 = read_keypoints("keypoints1")
    kp2 = read_keypoints("keypoints2")

    #alignment
    aligned_img = aligner_affine(img1, img2, kp1, kp2)
    cv2.imshow("Aligned Image", aligned_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Aligned Image")

    #frequency fusion
    result = frequency_hybrid_images(img1, aligned_img, D0=50, weight_low=0.5, weight_high=2)



    cv2.imwrite("frequency_hybrid.jpg", result)
    cv2.imshow("Frequency Hybrid", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #spatial fusion
    result2 = spatial_hybrid(img1, aligned_img)
    cv2.imwrite('spatial_hybrid.jpg',result2)
    cv2.imshow("Spatial Hybrid1", result2)
    cv2.waitKey(0)
    

main()







