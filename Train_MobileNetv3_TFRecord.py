import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from object_detection.utils import label_map_util

# Define the paths to the TFRecord files
train_tfrecord_path = 'C:/Dataset_TFRecord/train/Dressings.tfrecord'
val_tfrecord_path = 'C:/Dataset_TFRecord/valid/Dressings.tfrecord'
test_tfrecord_path = 'C:/Dataset_TFRecord/test/Dressings.tfrecord'
label_map_path = 'C:/Dataset_TFRecord/train/Dressings_label_map.pbtxt'

# Load the label map
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=4)
category_index = label_map_util.create_category_index(categories)

# Define the input shape of the images
input_shape = (224, 224, 3)

# Define the batch size and number of epochs for training
batch_size = 32
num_epochs = 100

# Define the function to parse the TFRecord example
def parse_tfrecord(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, input_shape[:2])
    image = tf.cast(image, tf.float32) / 255.
    label = tf.one_hot(example['image/class/label'], depth=4)
    return image, label

# Define the data generators for training, validation, and testing
train_data = tf.data.TFRecordDataset(train_tfrecord_path)
train_data = train_data.map(parse_tfrecord).shuffle(10000).batch(batch_size)

val_data = tf.data.TFRecordDataset(val_tfrecord_path)
val_data = val_data.map(parse_tfrecord).batch(batch_size)

test_data = tf.data.TFRecordDataset(test_tfrecord_path)
test_data = test_data.map(parse_tfrecord).batch(batch_size)

# Define the MobileNetV3Small model
base_model = keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False, weights='imagenet')
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=num_epochs, validation_data=val_data)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)

# Save the model
model.save('my_model.h5')
