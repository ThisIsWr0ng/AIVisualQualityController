import cv2
import os

# Image dimensions expected by the model.
input_size = (224, 224)

# Destination for the resized dataset.
output_folder = 'C:/Dataset_Label'

# Create the destination directory when necessary.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each class directory in the source dataset.
for foldername in os.listdir('C:/Dataset'):
    folder_path = os.path.join('C:/Dataset', foldername)
    if not os.path.isdir(folder_path):
        continue
    
    # Process every file in the current class directory.
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        # Load and resize the image to the model's input dimensions.
        image = cv2.imread(file_path)

        resized_image = cv2.resize(image, input_size)

        # Build a normalized blob for compatibility with the model pipeline.
        normalized_image = cv2.dnn.blobFromImage(resized_image, scalefactor=1/255.0, size=input_size, mean=(0,0,0), swapRB=True, crop=False)

        # Prefix the output filename with its class directory name.
        output_path = os.path.join(output_folder, f'{foldername}_{filename}')
        cv2.imwrite(output_path, resized_image)
