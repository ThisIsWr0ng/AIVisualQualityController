import cv2
import darknet

# Set the network configuration and weights files
config_file = 'custom-yolov4-tiny-detector'
weights_file = 'weights_v3.weights'
data_file = 'obj.names'

# Load the network
network, class_names, class_colors = darknet.load_network(config_file, data_file, weights_file)

# Set the width and height of the network input
width = 640
height = 640

# Open a video capture object for the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame to the network input size
    frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert the frame to Darknet format
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

    # Perform object detection on the frame
    detections = darknet.detect_image(network, class_names, darknet_image)

    # Draw the detections on the frame
    darknet.draw_boxes(detections, frame_resized, class_colors)

    # Show the frame with the detections
    cv2.imshow('YOLOv4-tiny Object Detection', frame_resized)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()