import cv2
import numpy as np
import tensorflow as tf
import time

# Load your trained model
model_path = 'path/to/your/trained/model.h5'
model = tf.keras.models.load_model(model_path)

# Label map
label_map = {0: 'Cut', 1: 'Dressing', 2: 'F_Body', 3: 'Red_T'}

def process_frame(frame, model, input_size):
    resized_frame = cv2.resize(frame, input_size)
    input_data = np.expand_dims(resized_frame, axis=0) / 255.0
    predictions = model.predict(input_data)
    return predictions

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

prev_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    current_time = time.time()
    fps = (1 / (current_time - prev_time)) *60
    prev_time = current_time
    
    # Process the frame and get the predictions
    predictions = process_frame(frame, model, (224, 224))
    
    # Assuming your model outputs both class predictions and bounding box coordinates
    class_predictions, bbox_predictions = predictions

    # Get the detected class and confidence
    predicted_class = np.argmax(class_predictions)
    confidence = np.max(class_predictions)

    # Get the bounding box coordinates
    y_min, x_min, y_max, x_max = bbox_predictions

    # Scale the bounding box coordinates to the original frame size
    height, width, _ = frame.shape
    x_min = int(x_min * width)
    x_max = int(x_max * width)
    y_min = int(y_min * height)
    y_max = int(y_max * height)

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the detected class and confidence on the frame
    text = f"{label_map[predicted_class]}: {confidence * 100:.2f}%"
    cv2.putText(frame, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display FPS on the frame
    fps_text = f"d/minute: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
