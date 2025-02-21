#spatial_fusion.py: contains function(s) for creating the hybrid image in the spatial domain using Gaussian/Laplacian pyramids.
#REFERENCE: https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html

import cv2
import numpy as np

#build gaussian pyr by repeatedly downsampling image and return list of imgs from original to smallest lvl.
def build_gaussian_pyramid(image, levels=6):
    pyramid = [image.copy()]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

#build laplacian pyr from gaussian pyr ^^^. Each lvl is difference between gaussian lvl and upsampled next lvl.
def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = [gaussian_pyramid[5]]
    for i in range(5, 0, -1):
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

#make the hybrid img by blending laplacing pyrs of img1 and 2. Use horizontal mask to blend this nicely 
def spatial_hybrid(image1, image2, blend_width=200):
    #build gauss pyrs for both images
    gp1 = build_gaussian_pyramid(image1, levels=6) #img 1
    gp2 = build_gaussian_pyramid(image2, levels=6) #img2
    
    #build Laplacian pyrs from the gauss pyrs
    lp1 = build_laplacian_pyramid(gp1)
    lp2 = build_laplacian_pyramid(gp2)
    
    #blend Laplacian levels using horizontal mask
    blended_levels = []
    alpha = 0.65 #change this for TESTING for image2 high-freq details
    for lap1, lap2 in zip(lp1, lp2):
        #determinne dimensions and channel count
        if len(lap1.shape) == 2:
            rows, cols = lap1.shape
            channels = 1
        else:
            rows, cols, channels = lap1.shape
        mid_col = cols // 2
        left_bound = max(0, mid_col - blend_width // 2)
        right_bound = min(cols, mid_col + blend_width // 2)
        region_width = right_bound - left_bound
        
        #create 1D horizontal mask, transitions from 1 to 0
        mask_1d = np.ones(cols, dtype=np.float32)
        mask_1d[left_bound:right_bound] = np.linspace(1, 0, region_width)
        mask_1d[right_bound:] = 0
        
        #expand 1D mask to the image size
        mask = np.tile(mask_1d, (rows, 1))
        if channels == 3:
            mask = cv2.merge([mask, mask, mask])
        
        #blend two Laplacian images using the mask and an alpha scaling for image2
        blended = lap1.astype(np.float32) * mask + lap2.astype(np.float32) * alpha * (1 - mask)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        blended_levels.append(blended)
    
    #construct, save to file, and return final blended image
    hybrid_image = reconstruct_image_from_pyramid(blended_levels)
    cv2.imwrite('output_images/spatial_hybrid.jpg', hybrid_image) #write spatial_hybrid image to file.
    return hybrid_image