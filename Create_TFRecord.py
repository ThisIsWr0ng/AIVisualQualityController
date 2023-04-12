import os
import csv
import tensorflow as tf
import pandas as pd
from PIL import Image

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tfrecord(images, labels, bboxes, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for image_path, label, bbox in zip(images, labels, bboxes):
            with open(image_path, 'rb') as img_file:
                img = img_file.read()
            image = Image.open(image_path)
            width, height = image.size
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': _bytes_feature(img),
                'image/object/class/label': _int64_feature(label),
                'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
                'image/object/bbox/xmin': _float_feature([bbox[0]]),
                'image/object/bbox/ymin': _float_feature([bbox[1]]),
                'image/object/bbox/xmax': _float_feature([bbox[2]]),
                'image/object/bbox/ymax': _float_feature([bbox[3]])
            }))
            writer.write(example.SerializeToString())

def read_images_and_labels_from_csv(csv_file):
    images = []
    labels = []
    bboxes = []

    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        img_path = os.path.join(os.path.dirname(csv_file), row['class'], row['filename'])
        label = label_map[row['class']]
        bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])

        images.append(img_path)
        labels.append(label)
        bboxes.append(bbox)

    return images, labels, bboxes

def create_tfrecords_from_folder(data_folder, train_output, val_output, test_output):
    def process_folder(folder, output_file):
        images, labels, bboxes = read_images_and_labels_from_csv(os.path.join(folder, '_annotations.csv'))
        create_tfrecord(images, labels, bboxes, output_file)

    train_folder = os.path.join(data_folder, 'train')
    val_folder = os.path.join(data_folder, 'valid')
    test_folder = os.path.join(data_folder, 'test')

    process_folder(train_folder, train_output)
    process_folder(val_folder, val_output)
    process_folder(test_folder, test_output)

if __name__ == "__main__":
    data_folder = "C:\Dataset_Tensorflow_v5"
    train_output = "C:/Dataset_Tensorflow_v5/train/train.tfrecord"
    val_output = "C:/Dataset_Tensorflow_v5/valid/val.tfrecord"
    test_output = "C:/Dataset_Tensorflow_v5/test/test.tfrecord"
    label_map = {'Cut': 0, 'Dressing': 1, 'F_Body': 2, 'Red_T': 3}


    create_tfrecords_from_folder(data_folder, train_output, val_output, test_output)

#create label map.txt   
def create_label_map(data_folder, output_file):
    train_folder = os.path.join(data_folder, 'train')
    class_names = [d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]
    
    with open(output_file, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

label_map_output = 'C:\Dataset_Tensorflow_v5/label_map.txt'

create_label_map(data_folder, label_map_output)