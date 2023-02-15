import cv2
import numpy as np
import matplotlib.pylab as plt

#cv2.namedWindow("preview")
url = "https://192.168.1.239:8080/video"# link to smartphone camera
webcam = 0 # 0 for laptop webcam / 1 for external webcam
vc = cv2.VideoCapture(url)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    vc = cv2.VideoCapture(webcam)
    rval, frame = vc.read()
    #rval = False

print(frame.shape)
height = frame.shape[0]
width = frame.shape[1]
region_of_interest_vertices = [
    (width/4, height/4),
    (width/0.25, height/4),
    (width/4, height/0.25),
    (width/0.25, height/0.25)
]

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



while rval:
    
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    #cropFrame = region_of_interest(frame,np.array([region_of_interest_vertices], np.int32),)
    #cv2.imwrite("frame.png", frame)
    #img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    cv2.imshow('edges', edges)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("Main", frame)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("Main")