import os
import cv2
import matplotlib.pyplot as plt

def imShow(path):
  #%matplotlib inline

  image = cv2.imread(path) 
  if image is None: print('Failed to load image from', path) 
  else:
    height, width = image.shape[:2]
    resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    #plt.rcParams['figure.figsize'] = [10, 5]
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()

  #/test has images that we can test our detector on

test_images = [f for f in os.listdir(r'C:\Dataset_YOLO\test')  if f.endswith('.jpg')]
import random
img_path = "C:\Dataset_YOLO\test" + random.choice(test_images)

weights = 'Weights_v1.weights'

#os.system(f'copy data/obj.names data/coco.names')
#os.system(f'darknet.exe detect cfg/custom-yolov4-tiny-detector.cfg {weights} {img_path} -dont-show')
os.system('c:\darknet\darknet.exe detector test cfg/custom-yolov4-tiny-detector.cfg {weights} {img_path} -dont-show')



imShow('darknet/predictions.jpg')