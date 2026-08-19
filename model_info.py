import tensorflow as tf
import sys
import cv2
import numpy as np
# Load the Keras model to inspect or convert it.
model = tf.keras.models.load_model('C:/Users/dawid/OneDrive/Documents/GitHub/AIVisualQualityController/model.h5')

# Optional: convert the model to TensorFlow Lite.
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open('path/to/save/model.tflite', 'wb') as f:
#     f.write(tflite_model)


# Optional: write the model summary to a text file.
# original_stdout = sys.stdout

# with open('model_summary.txt', 'w') as f:
#     sys.stdout = f
#     model.summary()

# sys.stdout = original_stdout
test_image_path = 'C:/Dataset_Tensorflow_v3/test/Cut/WIN_20230301_18_30_13_Pro_jpg.rf.02f496200631e77ed287a148184a3d1f.jpg'
test_image = cv2.imread(test_image_path)
resized_image = cv2.resize(test_image, (224, 224))
input_data = np.expand_dims(resized_image, axis=0) / 255.0

predictions = model.predict(input_data)
print(predictions)
