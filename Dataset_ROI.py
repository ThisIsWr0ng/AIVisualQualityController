import cv2
import numpy as np
import os

margin = 20
input_folder = "C:/Dataset"
output_folder = "C:/Dataset_ROI"

# Define the colors to be used for drawing contours and lines
green = (0, 255, 0)
red = (0, 0, 255)

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over each subfolder in the input folder
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    output_subfolder_path = os.path.join(output_folder, subfolder)
    if not os.path.exists(output_subfolder_path):
        os.makedirs(output_subfolder_path)

    # Loop over each image in the subfolder
    for filename in os.listdir(subfolder_path):
        filepath = os.path.join(subfolder_path, filename)
        output_path = os.path.join(output_subfolder_path, filename)
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding with a lower threshold of 100 and upper threshold of 255
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the original image and find the largest contour
        largest_contour = None
        for contour in contours:
            #cv2.drawContours(img, [contour], -1, green, 3)
            if largest_contour is None or cv2.contourArea(contour) > cv2.contourArea(largest_contour):
                largest_contour = contour

        if largest_contour is not None:
            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Make sure the ROI is square and add a margin of 20 pixels
            max_side = max(w, h)
            roi_x = max(0, x - (max_side - w) // 2 - margin)
            roi_y = max(0, y - (max_side - h) // 2 - margin)
            roi_w = min(max_side + 2 * margin, img.shape[1] - roi_x)
            roi_h = min(max_side + 2 * margin, img.shape[0] - roi_y)

            # Extract the ROI and save it to the output folder
            roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            cv2.imwrite(output_path, roi)
        else:
            print(f"No contour found for {filename}")

print("Done!")
