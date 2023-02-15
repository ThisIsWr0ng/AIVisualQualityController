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
object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold= 40)

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cropFrame = region_of_interest(frame,np.array([region_of_interest_vertices], np.int32),)
    #mask = object_detector.apply(frame)
    
    #mask = gray
    #_, mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
    #contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #try:
    #    for cnt in contours:
    #        #calc area to remove small elements
    #        area = cv2.contourArea(cnt)
    #        if area > 5000:
    #            #cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
    #            x, y, w, h = cv2.boundingRect(cnt)
    #            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 3)
    #except:   
    #    cv2.imshow("Main", frame) 

    
    edges = cv2.Canny(gray,250,400,apertureSize = 3)
    cv2.imshow('edges', edges)#Display Greyscale image
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