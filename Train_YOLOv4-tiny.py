import os

# Set the paths to the Darknet executable and the configuration and dataset files
darknet_path = r'C:\Darknet\darknet.exe'
config_file = 'cfg/custom-yolov4-tiny-detector_4class_416.cfg'
data_file = 'data/obj.data'

# Training from original weights
pretrained_weights = 'yolov4-tiny.conv.29'
#Training from previous best model
#pretrained_weights = 'Weights_v1.weights'
# Set the number of classes
num_classes = 4

# Set the batch size and number of iterations
batch_size = 64
max_batches = 10000

# Set the learning rate schedule
learning_rate = 0.00261
learning_rate_decay = 0.0005
learning_rate_decay_steps = '4800, 5400'

# Set the momentum
momentum = 0.9

# Set the path to the output directory
output_dir = 'data/out'

#Uncomment the lines to run training, mapping, or testing
# train the model
#os.system(f"cd C:\darknet")
#Train
#command = f"cd c:/darknet \n darknet.exe detector train {data_file} {config_file} {pretrained_weights} -map" #-dont_show -gpus 0 -batch {batch_size} -max_batches {max_batches} -learning_rate {learning_rate} -learning_rate_decay {learning_rate_decay} -learning_rate_decay_steps {learning_rate_decay_steps} -momentum {momentum} -out {output_dir}"
#Map
command = f"darknet.exe detector map {data_file} {config_file} backup/custom-yolov4-tiny-detector_4class_416_final.weights"
#Test
#command = f"darknet.exe detector test {data_file} {config_file} backup/custom-yolov4-tiny-detector_4class_final.weights -thresh 0.5"
#os.system(command)
print(command)
