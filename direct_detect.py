import cv2
import numpy as np

# Load YOLOv4-tiny with Darknet
net = cv2.dnn.readNet('c:\darknet\weights_v3.weights', 'C:\Darknet\cfg\yolov4-tiny.cfg')
classes = []
with open('C:\Darknet\data\obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
#print(classes)
#output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = net.getUnconnectedOutLayersNames()

# Initialize camera feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


while True:
    # Read frame from camera feed
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (640, 640))
    # Preprocess the frame for YOLOv4-tiny
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    #output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
 
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    # Process the outputs of YOLOv4-tiny
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                #print(class_ids)

    # Apply non-max suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indices)
 # Draw the boxes on the frame and show the result
    if len(indices) > 0:
        for i in indices.flatten():
            #print(i)
            x, y, w, h = boxes[i]
            # Get the detection corresponding to this index
            detection = layer_outputs[0][i]
            # Get the class ID and confidence from the detection
            class_id = np.argmax(detection[5:])
            confidence = detection[5:][class_id]
            print(class_id)
            label = str(classes[class_id])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
