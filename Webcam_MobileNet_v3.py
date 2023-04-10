import cv2
import numpy as np
import tensorflow as tf
import time

# Load your trained model
model_path = 'model.h5'
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
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Process the frame and get the predictions
    predictions = process_frame(frame, model, (224, 224))
    
    # Get the detected class and confidence
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display the detected class and confidence on the frame
    text = f"{label_map[predicted_class]}: {confidence * 100:.2f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display FPS on the frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()