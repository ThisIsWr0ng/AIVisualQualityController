import cv2
import time
import numpy as np
import onnxruntime as ort
import torchvision.transforms as T
import torch

# Load model
ort_session = ort.InferenceSession("Model/Yolo_weights_quant.onnx")


torch.backends.quantized.engine = 'qnnpack'
# Initialize camera
cap = cv2.VideoCapture(0)

# Define image pre-processing transforms
preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((416, 416)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():
    while True:
        # Read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # Preprocess input image
        input_tensor = preprocess(image)
        input_numpy = input_tensor.numpy()
        outputs = ort_session.run(None, {"input": [input_numpy]})
        print("Number of elements in outputs:", len(outputs))
        print("Outputs:", outputs)



        # Extract boxes, scores, and labels from the output
        boxes = outputs[0]
        scores = outputs[1]
        labels = outputs[2]

        # Draw bounding boxes on the original image
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5:  # Set a threshold for the confidence score
                x1, y1, x2, y2 = box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(image, str(label), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the camera feed with bounding boxes
        cv2.imshow("Camera Feed", image)

        # Log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


