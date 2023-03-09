import cv2
import numpy as np
import matplotlib.pylab as plt

webcam = 0 # 0 for laptop webcam / 1 for external webcam
vc = cv2.VideoCapture(webcam)
rval, frame = vc.read()

margin = 20 # adjust this value to change the size of the bounding box

while rval:
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,250,600,apertureSize = 3)

    # Extract ROI based on edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        x -= margin
        y -= margin
        w += 2*margin
        h += 2*margin
        roi = frame[max(y, 0):min(y+h, frame.shape[0]), max(x, 0):min(x+w, frame.shape[1])]
        cv2.imshow('ROI', roi)

    #cv2.imshow('edges', edges)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,2,minLineLength=0,maxLineGap=0)
    try:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),4)

    except:
        cv2.imshow("Main", frame)
    cv2.imshow("Main", frame)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("Main")
cv2.destroyWindow("ROI")