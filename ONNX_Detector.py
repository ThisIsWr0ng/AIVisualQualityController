import cv2
import numpy as np
import onnxruntime as ort

# Load YOLOv4-tiny ONNX model
ort_session = ort.InferenceSession('Model/Yolo_weights.onnx')

classes = []
with open('Model/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize camera feed
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera feed
    ret, frame = cap.read()
    frame = cv2.resize(frame, (416, 416))

    # Preprocess the frame for YOLOv4-tiny
    input_numpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_numpy = np.transpose(input_numpy, (2, 0, 1))
    input_numpy /= 255.0
    input_tensor = input_numpy[np.newaxis, :, :, :]

    # Run inference with ONNX Runtime
    outputs = ort_session.run(None, {"input": input_tensor})


    # Process the outputs of YOLOv4-tiny
    boxes = []
    confidences = []
    class_ids = []
    bbox_output = outputs[0].reshape(-1, 4)
    class_output = outputs[1].reshape(-1, 4)

    for detection, scores in zip(bbox_output, class_output):
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

    # Apply non-max suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the boxes on the frame and show the result
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            label = str(classes[class_id])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
