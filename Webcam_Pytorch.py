import cv2
import time
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models, transforms
# Load model and put it in eval mode
torch.backends.quantized.engine = 'qnnpack'
net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
net.eval()
net = torch.jit.script(net)

# Initialize camera
cap = cv2.VideoCapture(0)

# Define image pre-processing transforms
preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
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

        # Run model
        output = net([input_tensor])

        # Extract boxes, scores, and labels from the output
        losses, detections = output
        boxes = detections[0]['boxes'].cpu().numpy()
        scores = detections[0]['scores'].cpu().numpy()
        labels = detections[0]['labels'].cpu().numpy()

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
