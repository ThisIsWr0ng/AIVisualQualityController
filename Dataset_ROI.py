# Crop each image around the largest detected object contour.
import cv2
import numpy as np
import os

margin = 20
input_folder = r'C:\Dataset'
output_folder = r'C:\Dataset_ROI'

# Colors reserved for optional contour visualization.
green = (0, 255, 0)
red = (0, 0, 255)

# Create the output directory tree when necessary.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Preserve the source dataset's subdirectory structure.
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    output_subfolder_path = os.path.join(output_folder, subfolder)
    if not os.path.exists(output_subfolder_path):
        os.makedirs(output_subfolder_path)

    # Process each image in the current subdirectory.
    for filename in os.listdir(subfolder_path):
        filepath = os.path.join(subfolder_path, filename)
        output_path = os.path.join(output_subfolder_path, filename)
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to a binary mask.
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find the external contours in the binary mask.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the contour with the largest area.
        largest_contour = None
        for contour in contours:
            # Uncomment to visualize each candidate contour.
            # cv2.drawContours(img, [contour], -1, green, 3)
            if largest_contour is None or cv2.contourArea(contour) > cv2.contourArea(largest_contour):
                largest_contour = contour

        if largest_contour is not None:
            # Calculate the bounding box of the largest contour.
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Expand the box toward a square ROI and add the configured margin.
            max_side = max(w, h)
            roi_x = max(0, x - (max_side - w) // 2 - margin)
            roi_y = max(0, y - (max_side - h) // 2 - margin)
            roi_w = min(max_side + 2 * margin, img.shape[1] - roi_x)
            roi_h = min(max_side + 2 * margin, img.shape[0] - roi_y)

            # Extract and save the region of interest (ROI).
            roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            cv2.imwrite(output_path, roi)
        else:
            print(f"No contour found for {filename}")

print("Done!")
