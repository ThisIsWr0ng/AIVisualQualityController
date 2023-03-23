import cv2
import numpy as np
import matplotlib.pylab as plt

webcam = 0 # 0 for laptop webcam / 1 for external webcam
vc = cv2.VideoCapture(webcam)
rval, frame = vc.read()

canny_min_thresh = 400
canny_max_thresh = 600
aperture_size = 3
margin = 20

margin = 20 # size of the bounding box

while rval:
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding with a lower threshold of 100 and upper threshold of 255
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    edges = cv2.Canny(thresh, canny_min_thresh, canny_max_thresh, apertureSize=aperture_size)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Preview', edges)
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
        roi_w = min(max_side + 2 * margin, frame.shape[1] - roi_x)
        roi_h = min(max_side + 2 * margin, frame.shape[0] - roi_y)

        # Extract the ROI and save it to the output folder
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        cv2.imshow('ROI', roi)
    else:
        continue

    cv2.imshow("Main", frame)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("Main")
cv2.destroyWindow("ROI")



 
