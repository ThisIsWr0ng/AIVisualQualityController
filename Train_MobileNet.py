from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

# Image dimensions expected by the model.
input_size = (224, 224)

# Number of dataset classes.
num_classes = 4

# Number of samples in each training batch.
batch_size = 32

# Maximum number of training epochs.
num_epochs = 10

# Root directory of the labeled dataset.
dataset_dir = 'C:/Dataset_Label'

# Create augmented training data and normalized validation data.
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = data_generator.flow_from_directory(
    dataset_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = data_generator.flow_from_directory(
    dataset_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Load MobileNet with weights pre-trained on ImageNet.
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add pooling and classification layers.
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Combine the MobileNet backbone with the custom classifier.
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained backbone during initial training.
for layer in base_model.layers:
    layer.trainable = False

# Configure the optimizer, loss, and evaluation metric.
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the classifier on the labeled dataset.
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size)

# Save the trained model.
model.save('my_model.h5')
