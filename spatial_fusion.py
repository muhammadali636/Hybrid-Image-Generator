#spatial_fusion.py: contains function(s) for creating the hybrid image in the spatial domain using Gaussian/Laplacian pyramids.
#REFERENCE: https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html

import cv2
import numpy as np

#build gaussian pyr by repeatedly downsampling image and return list of imgs from original to smallest lvl.
def build_gaussian_pyramid(image, levels=4):
    pyramid = [image.copy()]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

#build laplacian pyr from gaussian pyr ^^^. Each lvl is difference between gaussian lvl and upsampled next lvl.
def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(len(gaussian_pyramid)-1, 0, -1):
        upsampled = cv2.pyrUp(gaussian_pyramid[i])
        #adjust size if needed to match previous lvl
        if upsampled.shape[:2] != gaussian_pyramid[i - 1].shape[:2]:
            upsampled = cv2.resize(upsampled, (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0]))
        lap_layer = cv2.subtract(gaussian_pyramid[i - 1], upsampled)
        laplacian_pyramid.append(lap_layer)
    return laplacian_pyramid

#reconstruct img by upsampling and  adding laplacian pyr levels.
def reconstruct_image_from_pyramid(pyramid):
    image = pyramid[0]
    for level in range(1, len(pyramid)):
        image = cv2.pyrUp(image)
        if image.shape[:2] != pyramid[level].shape[:2]:
            image = cv2.resize(image, (pyramid[level].shape[1], pyramid[level].shape[0]))
        image = cv2.add(image, pyramid[level])
    return image

#make the hybrid img by combining low frequencies of img1 with high frequencies of img2,
#so that the images appear differently when viewed from different distances.
#**we can play with number of levels and the alpha to adjust the spatial hybrid.**
def spatial_hybrid(image1, image2, alpha=0.35, levels=4):
    #build gauss pyr for img1
    gp1 = build_gaussian_pyramid(image1, levels=levels)
    #get low frequency component from img1 (smallest gaussian level) and upsample to original size
    low_image = gp1[-1]
    for _ in range(levels):
        low_image = cv2.pyrUp(low_image)
        if low_image.shape[:2] != image1.shape[:2]:
            low_image = cv2.resize(low_image, (image1.shape[1], image1.shape[0]))
    
    #build gauss pyr for img2 and then its laplacian pyr to extract high frequency details
    gp2 = build_gaussian_pyramid(image2, levels=levels)
    lp2 = build_laplacian_pyramid(gp2)
    high_image = reconstruct_image_from_pyramid(lp2)
    
    #combine low frequencies from img1 and high frequencies from img2 using weighted addition
    hybrid_image = cv2.addWeighted(low_image, 1.0, high_image, alpha, 0)
    cv2.imwrite('output_images/spatial_hybrid.jpg', hybrid_image) #write spatial_hybrid image to file.
    return hybrid_image
