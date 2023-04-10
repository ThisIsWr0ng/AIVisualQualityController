import os
import shutil
import pandas as pd

def split_dataset(data_dir, annotations_csv, output_dir):
    df = pd.read_csv(annotations_csv)

    for idx, row in df.iterrows():
        image_path = os.path.join(data_dir, row['filename'])
        class_name = row['class']
        class_dir = os.path.join(output_dir, class_name)

        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        shutil.copy(image_path, class_dir)

    # Copy the annotation file to the output directory
    shutil.copy(annotations_csv, output_dir)

data_folders = ['train', 'valid', 'test']
base_data_folder = 'C:\Dataset_Tensorflow_v2'
output_base_folder = 'C:\Dataset_Tensorflow_v3'

for data_folder in data_folders:
    input_data_folder = os.path.join(base_data_folder, data_folder)
    output_data_folder = os.path.join(output_base_folder, data_folder)
    annotations_file = os.path.join(input_data_folder, f'_annotations.csv')

    split_dataset(input_data_folder, annotations_file, output_data_folder)
