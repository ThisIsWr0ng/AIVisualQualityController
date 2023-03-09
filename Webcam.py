import cv2
import numpy as np
import matplotlib.pylab as plt

webcam = 0 # 0 for laptop webcam / 1 for external webcam
vc = cv2.VideoCapture(webcam)
rval, frame = vc.read()

canny_min_thresh = 250
canny_max_thresh = 600
aperture_size = 3

margin = 20 # adjust this value to change the size of the bounding box

while rval:
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, canny_min_thresh, canny_max_thresh, apertureSize=aperture_size)

    # Extract ROI based on edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        x -= margin
        y -= margin
        w += 2*margin
        h += 2*margin

        # Make ROI always a square
        roi_size = max(w, h)
        x_center = x + w // 2
        y_center = y + h // 2
        x = x_center - roi_size // 2
        y = y_center - roi_size // 2
        w = h = roi_size

        roi = frame[max(y, 0):min(y+h, frame.shape[0]), max(x, 0):min(x+w, frame.shape[1])]
        cv2.imshow('ROI', roi)


    cv2.imshow("Main", frame)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("Main")
cv2.destroyWindow("ROI")
