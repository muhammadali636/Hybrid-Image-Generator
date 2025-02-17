
#alignment.py
import cv2
import numpy as np


#Contains the function(s) to read keypoints and perform either rigid (affine) or non-rigid registration. USE NONRIGID
#Must produce and save aligned.jpg (aligned image2.jpg).

#read images
def read_image(image1: str ,image2: str):
    img1 = cv2.imread(image1)     #cv2.imread(image1, cv.IMREAD_GRAYSCALE) for grayscale.     #DIMENSIONS: print('image height is', image1.shape[0]), print('image width is',image1.shape[1])
    img2 = cv2.imread(image2)

    if img1 is None or img2 is None:
        print("can't read {image1} or {image2}")
        return None
    return img1, img2, 

    #IMAGE TESTING
    #cv2.imshow('image1', img1) #show
    #cv2.waitKey(0)  #wait for a key press indefinitely to exit
    #cv2.destroyAllWindows()

def read_keypoints(filename):
    with open(filename, 'r') as file:
        content = file.read().strip()
    
    # If file is empty, return an empty array
    if not content:
        return np.array([], dtype=np.float32)
    
    # Remove the outer square brackets if present
    if content[0] == '[' and content[-1] == ']':
        content = content[1:-1]
    
    # Split the content by the delimiter between keypoints.
    # This assumes keypoints are separated by ")," 
    kp_strings = content.split("),")
    
    kp_list = []
    for kp_str in kp_strings:
        # Remove any remaining parentheses and whitespace
        kp_str = kp_str.replace("(", "").replace(")", "").strip()
        if kp_str:
            parts = kp_str.split(',')
            # Expecting two parts for x and y
            x = float(parts[0].strip())
            y = float(parts[1].strip())
            kp_list.append((x, y))
    
    return np.array(kp_list, dtype=np.float32)



def aligner(image1, image2, kp1, kp2):
    if image1 is None or image2 is None:
        print("Can't read image1 or image2")
        return None

    for key in kp1:
        print(key)
    # If there are no matches or insufficient matches, simply resize image2 to match image1
    '''
    if matches is None or len(matches) < 3:
        print("Not enough matches. Resizing image2 to match image1.")
        image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        cv2.imwrite('aligned.jpg', image2_resized)
        return image2_resized
    '''
  