import cv2

cv2.namedWindow("previe")
url = "https://192.168.1.239:8080/video"
webcam = 0 # 0 for laptop webcam / 1 for external webcam
vc = cv2.VideoCapture(url)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    vc = cv2.VideoCapture(webcam)
    rval, frame = vc.read()
    #rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")