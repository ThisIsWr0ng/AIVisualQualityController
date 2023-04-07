import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV3Small
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model

# Define the path to the directory containing the dataset
data_dir = 'C:\Dataset_Tensorflow'

# Define the input shape of the images
input_shape = (640, 640, 3) # add an additional dimension for the channels

# Define the batch size and number of epochs for training
batch_size = 32
num_epochs = 100

# Define the data generators for training, validation, and testing
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

# Define the MobileNetV3Small model
inputs = Input(shape=input_shape)
x = MobileNetV3Small(input_tensor=inputs, classes=1000, include_top=False, weights='imagenet')(inputs)
x = GlobalAveragePooling2D()(x) # reduce the spatial dimensions of the feature maps
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data,
                    epochs=num_epochs,
                    validation_data=val_data)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)

# Save the model
model.save('my_model.h5')
