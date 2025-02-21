#keypointer.py gives me the keypoints of my images. use 2 separate terminals and do the keypointing side by side. 
import cv2

#store keypoints
keypoints = []
inputted_image = input("Enter image file: ")

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Selected point: ({x}, {y})")
        keypoints.append((x, y))
        # Draw a small circle at the clicked point for visual feedback
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", img)

#load image
img = cv2.imread(inputted_image)
if img is None:
    print("Error: Image not found!")
    exit()

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

print("Click on the image to select keypoints. Press any key when done...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Collected keypoints for {inputted_image}:", keypoints)

#determine output file based on input file name (keypoints 1 or 2)
if "1" in inputted_image:
    output_filename = "keypoints1"
elif "2" in inputted_image:
    output_filename = "keypoints2"

#write the keypoints to the file (either keypoints 1 or 2)
with open(output_filename, "w") as file:
    for point in keypoints:
        file.write(f"{point}\n")

print(f"Keypoints saved to {output_filename}")
