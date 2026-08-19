import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV3Small
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model

# Root directory of the image dataset.
data_dir = 'C:\Dataset_Tensorflow'

# Model input dimensions, including the RGB channels.
input_shape = (640, 640, 3)

# Training parameters.
batch_size = 32
num_epochs = 100

# Create augmented training data and normalized validation/test data.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(data_dir + '/train',
                                                target_size=input_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='categorical')

val_data = val_datagen.flow_from_directory(data_dir + '/valid',
                                            target_size=input_shape[:2],
                                            batch_size=batch_size,
                                            class_mode='categorical')

test_data = test_datagen.flow_from_directory(data_dir + '/test',
                                              target_size=input_shape[:2],
                                              batch_size=batch_size,
                                              class_mode='categorical')

# Build a MobileNetV3Small classifier.
inputs = Input(shape=input_shape)
x = MobileNetV3Small(input_tensor=inputs, classes=1000, include_top=False, weights='imagenet')(inputs)
x = GlobalAveragePooling2D()(x)  # Reduce each feature map to a single value.
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# Configure the optimizer, loss, and evaluation metric.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train with validation after every epoch.
history = model.fit(train_data,
                    epochs=num_epochs,
                    validation_data=val_data)

# Evaluate the final model on the held-out test set.
test_loss, test_acc = model.evaluate(test_data)

# Save the trained model.
model.save('my_model.h5')
