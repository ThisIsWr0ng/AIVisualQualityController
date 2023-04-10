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
    top_two_indices = np.argpartition(predictions[0], -2)[-2:]
    top_two_probs = predictions[0][top_two_indices]
    top_two_labels = [label_map[i] for i in top_two_indices]
    return top_two_labels, top_two_probs

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

    # Find the largest contour
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 400, 600, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    for contour in contours:
        if largest_contour is None or cv2.contourArea(contour) > cv2.contourArea(largest_contour):
            largest_contour = contour

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        max_side = max(w, h)
        margin = 20
        roi_x = max(0, x - (max_side - w) // 2 - margin)
        roi_y = max(0, y - (max_side - h) // 2 - margin)
        roi_w = min(max_side + 2 * margin, frame.shape[1] - roi_x)
        roi_h = min(max_side + 2 * margin, frame.shape[0] - roi_y)

        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        cv2.imshow('ROI', roi)

            # Process the ROI and get the predictions
    top_two_labels, top_two_probs = process_frame(roi, model, (224, 224))

    # Display the top two detected classes and confidences on the ROI
    text1 = f"{top_two_labels[0]}: {top_two_probs[0] * 100:.2f}%"
    text2 = f"{top_two_labels[1]}: {top_two_probs[1] * 100:.2f}%"
    cv2.putText(roi, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(roi, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display FPS on the frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(roi, fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the ROI
    cv2.imshow('Object Detection', roi)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()