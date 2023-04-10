import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord(images, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for img_path, label in zip(images, labels):
            img = Image.open(img_path)
            width, height = img.size
            img_bytes = img.tobytes()

            feature = {
                'image/encoded': _bytes_feature(img_bytes),
                'image/object/class/label': _int64_feature(label),
                'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def create_tfrecords_from_folder(data_folder, train_output, val_output, test_output, val_split=0.1, test_split=0.1):
    class_names = os.listdir(data_folder)
    images, labels = [], []
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(data_folder, class_name)
        for image_name in os.listdir(class_folder):
            images.append(os.path.join(class_folder, image_name))
            labels.append(class_index)

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_split, stratify=labels)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=val_split, stratify=train_labels)

    create_tfrecord(train_images, train_labels, train_output)
    create_tfrecord(val_images, val_labels, val_output)
    create_tfrecord(test_images, test_labels, test_output)

data_folder = 'C:\Dataset_Tensorflow_v3'
train_output = 'train.tfrecord'
val_output = 'val.tfrecord'
test_output = 'test.tfrecord'

create_tfrecords_from_folder(data_folder, train_output, val_output, test_output)
