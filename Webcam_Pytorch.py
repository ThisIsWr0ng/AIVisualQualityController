import cv2
import time
import numpy as np
import onnxruntime as ort
import torchvision.transforms as T
import torch
from torchvision.ops import nms

# Load model
ort_session = ort.InferenceSession("Model/Yolo_weights_quant.onnx")

# Read class names from file
with open("Model/obj.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

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

# Choose a object tracker type
tracker_type = 'KCF'

# Function to create a tracker based on the tracker type
def create_tracker():
    if tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        return cv2.TrackerTLD_create()
    elif tracker_type == 'MOSSE':
        return cv2.TrackerMOSSE_create()
    else:
        raise ValueError("Invalid tracker type")

# Initialize trackers for each detected object
trackers = []
frame_counter = 0
detection_interval = 10 #how many frames to skip between detection
#detection loop
with torch.no_grad():
    while True:
        frame_counter += 1
        is_detection_frame = frame_counter % detection_interval == 0
        # Read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")
        if is_detection_frame:


            # Preprocess input image
            input_tensor = preprocess(image)
            input_numpy = input_tensor.numpy()
            outputs = ort_session.run(None, {"input": [input_numpy]})

            # Extract boxes, scores, and labels from the output
            boxes = outputs[0].reshape(-1, 4)
            scores = outputs[1].squeeze(0)


            #print("boxes: ", boxes, "Scores: ", scores)

            # Convert boxes and scores to PyTorch tensors
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            # Apply NMS
            nms_threshold = 0.5
            keep_indices = []
            for class_index in range(scores.shape[1]):
                class_scores = scores_tensor[:, class_index]
                class_indices = nms(boxes_tensor, class_scores, nms_threshold)
                keep_indices.extend([(i, class_index) for i in class_indices])

            # Filter boxes, scores, and labels using the keep_indices
            filtered_boxes = np.array([boxes[i] for i, c in keep_indices if c == 1])  # Only class 1
            filtered_scores = np.array([scores[i, c] for i, c in keep_indices if c == 1])  # Only class 1
            filtered_labels = np.array([c for _, c in keep_indices if c == 1])  # Only class 1

            height, width, _ = image.shape
            # Draw bounding boxes on the original image
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                if np.max(score) > 0.5:
                    
                    pixel_boxes = []
                    for box in filtered_boxes:
                        x1, y1, x2, y2 = box * np.array([width, height, width, height])
                        pixel_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

                    print("box: ",box)
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(image, f"{class_names[label]} {np.max(score):.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Initialize a tracker for each detected object
            trackers = [create_tracker() for _ in range(len(pixel_boxes))]
            for tracker, box in zip(trackers, pixel_boxes):
                tracker.init(image, tuple(box))
        else:
            # Update the trackers
            for tracker in trackers:
                success, box = tracker.update(image)
                if success:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
        # Calculate FPS
        frame_count += 1
        now = time.time()
        fps = frame_count / (now - last_logged)

        # Display FPS on the image
        cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the camera feed with bounding boxes
        cv2.imshow("Camera Feed", image)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

