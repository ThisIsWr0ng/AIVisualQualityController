import tensorflow as tf
from keras.applications import MobileNetV2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard

# Custom load_dataset function
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
num_epochs = 100
train_tfrecords = "C:/Dataset_Tensorflow_v4/train/train.tfrecord"
val_tfrecords = "C:/Dataset_Tensorflow_v4/valid/val.tfrecord"

# Load datasets
train_data = load_dataset(train_tfrecords, input_shape, batch_size, num_classes)
val_data = load_dataset(val_tfrecords, input_shape, batch_size, num_classes)

# Create and compile the model
base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
x = tf.keras.layers.Dense(num_classes, activation='softmax')(base_model.output)
model = tf.keras.Model(inputs=base_model.inputs, outputs=x)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

#Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True, start_from_epoch=20)
#tensorboard for logging
tensorboard = TensorBoard(log_dir='logs')
#tensorboard --C:\Users\dawid\OneDrive\Documents\GitHub\AIVisualQualityController\logs logs
# Train the model
history = model.fit(train_data, epochs=num_epochs, validation_data=val_data, callbacks=[early_stopping, tensorboard])
model.save('model_mobilev2_v2.h5')
