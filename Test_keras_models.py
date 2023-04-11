import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model
model = load_model('model_mobilev2_v2.h5')

# Define test dataset file
test_tfrecords = "C:/Dataset_Tensorflow_v4/test/test.tfrecord"
def load_dataset(tfrecords, input_shape, batch_size, num_classes):
    def parse_tfrecord(example_proto):
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
        image = tf.image.resize(image, input_shape[:2])
        image = tf.cast(image, tf.float32) / 255.

        label = example['image/object/class/label']
        if label.shape[0] == 0:
            label = tf.constant([0], dtype=tf.int64)
        else:
            label = label.values[0]
        label = tf.one_hot(label, depth=num_classes)

        return image, label

    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def read_label_map(label_map_file):
    with open(label_map_file, 'r') as f:
        lines = f.readlines()
    label_map = {i: class_name.strip() for i, class_name in enumerate(lines)}
    return label_map

label_map_file = 'C:\Dataset_Tensorflow_v4/label_map.txt'
label_map = read_label_map(label_map_file)

# Define parameters
input_shape = (416, 416, 3)
num_classes = len(label_map)
batch_size = 32


# Load test dataset

test_data = load_dataset(test_tfrecords, input_shape, batch_size, num_classes)

# Get true labels from the test dataset
y_true = []
for _, labels in test_data.unbatch():
    y_true.append(np.argmax(labels.numpy()))

y_true = np.array(y_true)

# Get model predictions on the test dataset
y_pred = np.argmax(model.predict(test_data), axis=1)

# Calculate and print the classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(label_map.values())))

# Calculate and print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
