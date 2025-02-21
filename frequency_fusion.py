#frequency_fusion.py: blending of high-frequency details from one image with low-frequency components from the other
#REFERENCES: https://www.geeksforgeeks.org/creating-hybrid-images-using-opencv-library-python/ , https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html

import cv2
import numpy as np
from math import sqrt, exp

#euclid distance between two points 
def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

#low pass: returns gauss low-pass filter for an image 
def gauss_low_pass(cutoff_frequency, image_shape):
    rows, cols = image_shape[:2]
    center_row, center_col = rows / 2, cols / 2
    y_indices, x_indices = np.ogrid[:rows, :cols]
    squared_distance_from_center = (y_indices - center_row) ** 2 + (x_indices - center_col) ** 2
    return np.exp(-squared_distance_from_center / (2 * cutoff_frequency ** 2))

#high pass: returns a gauss high-pass filter for an image 
def gauss_high_pass(cutoff_frequency, image_shape):
    rows, cols = image_shape[:2]
    center_row, center_col = rows / 2, cols / 2
    y_indices, x_indices = np.ogrid[:rows, :cols]
    squared_distance_from_center = (y_indices - center_row) ** 2 + (x_indices - center_col) ** 2
    return 1 - np.exp(-squared_distance_from_center / (2 * cutoff_frequency ** 2))

#make hybrid image by combining low frequencies from image1 and high frequencies from image2
def frequency_hybrid_images(image1, image2, cutoff_frequency=50): #cutoff_frequency = D0. 
    #fourier transform of image1 and shift zero frequency component to center (convert from spatial domain to frequency domain, center makes freq filter easier)
    fft_image1 = np.fft.fft2(image1) 
    fft_shifted_image1 = np.fft.fftshift(fft_image1)
    low_frequency_filter = gauss_low_pass(cutoff_frequency, image1.shape)   #high pass filter
    filtered_fft_low = fft_shifted_image1 * low_frequency_filter    #apply filter by multiplying it.
    inverse_fft_shifted_low = np.fft.ifftshift(filtered_fft_low)    #shift freqnecies to original position
    inverse_low_frequency_component = np.fft.ifft2(inverse_fft_shifted_low) #inverse fourier to convert back to spatial --> low freq component of image1.
    
    #same thing above but for image2 and high frequency. 
    fft_image2 = np.fft.fft2(image2)
    fft_shifted_image2 = np.fft.fftshift(fft_image2)
    high_frequency_filter = gauss_high_pass(cutoff_frequency, image2.shape) 
    filtered_fft_high = fft_shifted_image2 * high_frequency_filter 
    inverse_fft_shifted_high = np.fft.ifftshift(filtered_fft_high) 
    inverse_high_frequency_component = np.fft.ifft2(inverse_fft_shifted_high) 
    
    #combine low and high frequency components; weights can be adjusted to control each contribution
    weight_low = 0.6
    weight_high = 1.8
    hybrid_image = weight_low * np.abs(inverse_low_frequency_component) + weight_high * np.abs(inverse_high_frequency_component)
    
    #normalize hybrid image to the range 0 to 255 (displays require pixels in this range)
    normalized_hybrid_image = cv2.normalize(hybrid_image, None, 0, 255, cv2.NORM_MINMAX)
    hybrid_image_uint8 = normalized_hybrid_image.astype(np.uint8)
    cv2.imwrite('output_images/frequency_hybrid.jpg', hybrid_image_uint8) #write frequency_hybrid to file.
    return hybrid_image_uint8
