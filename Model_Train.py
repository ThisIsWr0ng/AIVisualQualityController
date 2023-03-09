from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

# Set input size of the model
input_size = (224, 224)

# Set number of classes
num_classes = 4

# Set batch size
batch_size = 32

# Set number of epochs
num_epochs = 10

# Set directory of labeled dataset
dataset_dir = 'C:/Dataset_Label'

# Create data generators for training and validation sets
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

# Load MobileNet model pre-trained on ImageNet dataset
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add global average pooling layer and output layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create new model with MobileNet base and custom output layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers of MobileNet base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the labeled dataset
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size)

# Save the trained model
model.save('my_model.h5')
