import tensorflow as tf
import numpy as np
import os
def parse_tfrecord(example_proto):
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    
    # decode the JPEG-encoded image
    image = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)
    
    # normalize pixel values to [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # extract bounding box coordinates and class label
    label = parsed_features['image/object/class/label']
    xmin = parsed_features['image/object/bbox/xmin']
    ymin = parsed_features['image/object/bbox/ymin']
    xmax = parsed_features['image/object/bbox/xmax']
    ymax = parsed_features['image/object/bbox/ymax']
    bbox = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    
    return image, {'bbox': bbox, 'classes': label}

# Define parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
LEARNING_RATE = 0.001
EPOCHS = 10
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 20

# Define the model
def create_model():
    base_model = tf.keras.applications.MobileNetV3Small(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

# Define the loss function
loss_object = tf.keras.losses.CategoricalCrossentropy()

# Define the metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Define the train step
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Define the validation step
@tf.function
def val_step(images, labels):
    predictions = model(images, training=False)
    v_loss = loss_object(labels, predictions)
    val_loss(v_loss)
    val_accuracy(labels, predictions)

# Load the dataset
train_dataset = tf.data.TFRecordDataset("C:/Dataset_Tensorflow_v2/train/Dressings.tfrecord")
train_dataset = train_dataset.map(parse_tfrecord)
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.TFRecordDataset("C:/Dataset_Tensorflow_v2/valid/Dressings.tfrecord")
val_dataset = val_dataset.map(parse_tfrecord)
val_dataset = val_dataset.shuffle(buffer_size=10000)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Create the model
model = create_model()

# Start training
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()

    for step, (images, labels) in enumerate(train_dataset):
        train_step(images, labels)

        if step == STEPS_PER_EPOCH:
            break

    for step, (images, labels) in enumerate(val_dataset):
        val_step(images, labels)

        if step == VALIDATION_STEPS:
            break

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          val_loss.result(),
                          val_accuracy.result()*100))
