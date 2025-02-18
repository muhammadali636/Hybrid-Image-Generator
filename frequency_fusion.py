# frequency_fusion.py
#https://www.geeksforgeeks.org/creating-hybrid-images-using-opencv-library-python/
#https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html


import cv2
import numpy as np
from math import sqrt, exp



#find fourier transformation of both images and apply zero component center shifting
#extract low frequency and high freqnecy component
#get image of low frequency and high frequency component using inverse fourier transformations
#combine spatial domain of the low pass filtered image and high-pass filtered image by adding magniteudes elementwise.


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

#low frequency extraction
def gaussianLP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for i in range(rows):
        for j in range(cols):
            base[i, j] = np.exp(-distance((i, j), center)**2 / (2 * D0**2))
    return base

#high freqnecy extraction, D0 is cutoff frequency
def gaussianHP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for i in range(rows):
        for j in range(cols):
            base[i, j] = 1 - np.exp(-distance((i, j), center)**2 / (2 * D0**2))
    return base

#function to generate hybrid image
def frequency_hybrid_images(image1, image2, D0=50, weight_low=0.5, weight_high=1):
    # Process image1 (low frequencies)
    original1 = np.fft.fft2(image1)                   # Get the Fourier transform of image1
    center1 = np.fft.fftshift(original1)              # Center the zero frequency
    LowPassCenter = center1 * gaussianLP(D0, image1.shape[:2])
    LowPass = np.fft.ifftshift(LowPassCenter)
    inv_LowPass = np.fft.ifft2(LowPass)               # Inverse FFT to get low-frequency image

    # Process image2 (high frequencies)
    original2 = np.fft.fft2(image2)                   # Get the Fourier transform of image2
    center2 = np.fft.fftshift(original2)              # Center the zero frequency
    HighPassCenter = center2 * gaussianHP(D0, image2.shape[:2])
    HighPass = np.fft.ifftshift(HighPassCenter)
    inv_HighPass = np.fft.ifft2(HighPass)             # Inverse FFT to get high-frequency image

    #combine with specified weights
    hybrid = weight_low * np.abs(inv_LowPass) + weight_high * np.abs(inv_HighPass)
    
    #normalize the result to 0-255 and convert to uint8
    hybrid = np.uint8(255 * (hybrid - hybrid.min()) / (hybrid.max() - hybrid.min()))
    return hybrid