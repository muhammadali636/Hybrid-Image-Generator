# spatial_fusion.py
# Contains function(s) for creating the hybrid image in the spatial domain using Gaussian/Laplacian pyramids.
#https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html
import cv2
import numpy as np
def spatial_hybrid(img1, img2, blend_width=200):
    # Copy input images
    A = img1.copy()
    B = img2.copy()
    
    # Generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    
    # Generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    
    #generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        if GE.shape[:2] != gpA[i-1].shape[:2]:
            GE = cv2.resize(GE, (gpA[i-1].shape[1], gpA[i-1].shape[0]))
        L = cv2.subtract(gpA[i-1], GE)
        lpA.append(L)
    
    #generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        if GE.shape[:2] != gpB[i-1].shape[:2]:
            GE = cv2.resize(GE, (gpB[i-1].shape[1], gpB[i-1].shape[0]))
        L = cv2.subtract(gpB[i-1], GE)
        lpB.append(L)
    
    #blend each level with a soft transition instead of a hard split.
    LS = []
    for la, lb in zip(lpA, lpB):
        #check if the image is grayscale or color
        if len(la.shape) == 2:
            rows, cols = la.shape
            ch = 1
        else:
            rows, cols, ch = la.shape
        
        mid = cols // 2
        
        #make sure the blend region is within image bounds:
        left_bound = max(0, mid - blend_width // 2)
        right_bound = min(cols, mid + blend_width // 2)
        region_width = right_bound - left_bound  # actual blend region width
        
        #create a 1D horizontal mask for this level:
        mask_1d = np.ones(cols, dtype=np.float32)
        # Use linspace over the valid blend region
        mask_1d[left_bound:right_bound] = np.linspace(1, 0, region_width)
        #set to 0 after the blend region (this is usually already 1 if not overwritten)
        mask_1d[right_bound:] = 0
        
        #expand mask to 2D.
        mask = np.tile(mask_1d, (rows, 1))
        #if color, expand mask to 3 channels.
        if ch == 3:
            mask = cv2.merge([mask, mask, mask])
        
        #blend the two Laplacian images for this level
        la_f = la.astype(np.float32)
        lb_f = lb.astype(np.float32)
        ls = la_f * mask + lb_f * (1 - mask)
        ls = np.clip(ls, 0, 255).astype(np.uint8)
        LS.append(ls)
    
    #reconstruct the final blended image from the pyramid
    ls_ = LS[0]
    for i in range(1, 6):
        ls_ = cv2.pyrUp(ls_)
        if ls_.shape[:2] != LS[i].shape[:2]:
            ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(ls_, LS[i])
    
    return ls_