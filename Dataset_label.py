import cv2
import os

# Set input size of the model
input_size = (224, 224)

# Set output folder for labeled dataset
output_folder = 'C:/Dataset_Label'

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over all folders in dataset
for foldername in os.listdir('C:/Dataset'):
    folder_path = os.path.join('C:/Dataset', foldername)
    if not os.path.isdir(folder_path):
        continue
    
    # Loop over all images in folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        # Load image
        image = cv2.imread(file_path)

        # Resize image to match input size of the model
        resized_image = cv2.resize(image, input_size)

        # Normalize image
        normalized_image = cv2.dnn.blobFromImage(resized_image, scalefactor=1/255.0, size=input_size, mean=(0,0,0), swapRB=True, crop=False)

        # Save labeled image to output folder
        output_path = os.path.join(output_folder, f'{foldername}_{filename}')
        cv2.imwrite(output_path, resized_image)
