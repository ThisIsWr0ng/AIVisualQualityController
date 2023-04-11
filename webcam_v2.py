import cv2
import numpy as np
import tensorflow as tf
import time
from keras.utils import custom_object_scope

def class_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def bbox_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load your trained model
model_path = 'model_mobilev2_v3.h5'
with custom_object_scope({'class_loss': class_loss, 'bbox_loss': bbox_loss}):
    model = tf.keras.models.load_model(model_path)

# Label map
label_map = {0: 'Cut', 1: 'Dressing', 2: 'F_Body', 3: 'Red_T'}

# Preallocate memory for input_data
input_data = np.empty((1, 224, 224, 3), dtype=np.float32)

def process_frame(frame, model, input_data):
    resized_frame = cv2.resize(frame, input_data.shape[1:3])
    np.copyto(input_data, resized_frame[np.newaxis] / 255.0)
    predictions, bbox_preds = model.predict(input_data)
    
    top_index = np.argmax(predictions)
    top_label = label_map[top_index]
    top_prob = predictions[0][top_index]
    predictions[0][top_index] = 0
    
    second_index = np.argmax(predictions)
    second_label = label_map[second_index]
    second_prob = predictions[0][second_index]
    
    return [(top_label, top_prob), (second_label, second_prob)], bbox_preds[0]

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
        cv2.imshow('camera', frame)

        # Process the ROI and get the predictions and bounding boxes
        top_two_preds, bbox_pred = process_frame(roi, model, input_data)

        # Display the top two detected classes and confidences on the ROI
        for i, (label, prob) in enumerate(top_two_preds):
            text = f"{label}: {prob * 100:.2f}%"
            cv2.putText(roi, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw predicted bounding boxes on the ROI
        bbox_xmin = int(bbox_pred[0] * roi_w)
        bbox_ymin = int(bbox_pred[1] * roi_h)
        bbox_xmax = int(bbox_pred[2] * roi_w)
        bbox_ymax = int(bbox_pred[3] * roi_h)

        cv2.rectangle(roi, (bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax), (0, 255, 0), 2)

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