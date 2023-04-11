import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

# Load the trained model
model = load_model('model_mobilev2_v3.h5')

# Define test dataset file
test_tfrecords = "C:/Dataset_Tensorflow_v5/test/test.tfrecord"

# ... (keep the load_dataset and read_label_map functions here)

# Define parameters
input_shape = (224, 224, 3)
num_classes = len(label_map)
batch_size = 32

# Load test dataset
test_data = load_dataset(test_tfrecords, input_shape, batch_size, num_classes)

# Get true labels and bounding boxes from the test dataset
y_true_labels = []
y_true_bboxes = []
for _, (labels, bboxes) in test_data.unbatch():
    y_true_labels.append(np.argmax(labels.numpy()))
    y_true_bboxes.append(bboxes.numpy())

y_true_labels = np.array(y_true_labels)
y_true_bboxes = np.stack(y_true_bboxes)

# Get model predictions on the test dataset
predictions = model.predict(test_data)
y_pred_labels = np.argmax(predictions[0], axis=1)
y_pred_bboxes = np.vstack(predictions[1])

# Calculate and print the classification report
print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=list(label_map.values())))

# Calculate and print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true_labels, y_pred_labels))

# Calculate and print the mean squared error for bounding box predictions
print("Bounding Box Mean Squared Error:")
print(mean_squared_error(y_true_bboxes, y_pred_bboxes))
