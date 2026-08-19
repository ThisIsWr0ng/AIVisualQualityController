import cv2
import numpy as np
import matplotlib.pylab as plt

webcam = 0  # Use 0 for the built-in camera or 1 for an external camera.
vc = cv2.VideoCapture(webcam)
rval, frame = vc.read()

canny_min_thresh = 400
canny_max_thresh = 600
aperture_size = 3
margin = 20

margin = 20  # Padding around the detected object, in pixels.

while rval:
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to a binary mask.
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Find external contours in the mask.
    edges = cv2.Canny(thresh, canny_min_thresh, canny_max_thresh, apertureSize=aperture_size)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Preview', edges)
    # Select the contour with the largest area.
    largest_contour = None
    for contour in contours:
        # Uncomment to visualize each candidate contour.
        # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
        if largest_contour is None or cv2.contourArea(contour) > cv2.contourArea(largest_contour):
                largest_contour = contour

    if largest_contour is not None:
        # Calculate the bounding box of the largest contour.
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Expand the box toward a square ROI and add the configured margin.
        max_side = max(w, h)
        roi_x = max(0, x - (max_side - w) // 2 - margin)
        roi_y = max(0, y - (max_side - h) // 2 - margin)
        roi_w = min(max_side + 2 * margin, frame.shape[1] - roi_x)
        roi_h = min(max_side + 2 * margin, frame.shape[0] - roi_y)

        # Extract and display the region of interest (ROI).
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        cv2.imshow('ROI', roi)
    else:
        continue

    cv2.imshow("Main", frame)
    if key == 27:  # Press Esc to exit.
        break

vc.release()
cv2.destroyWindow("Main")
cv2.destroyWindow("ROI")



 
