import os

# Darknet executable, model configuration, and dataset metadata.
darknet_path = r'C:\Darknet\darknet.exe'
config_file = 'cfg/custom-yolov4-tiny-detector_4class_416.cfg'
data_file = 'data/obj.data'

# Start from the original pre-trained weights.
pretrained_weights = 'yolov4-tiny.conv.29'
# Alternatively, resume from the best weights of an earlier run.
# pretrained_weights = 'Weights_v1.weights'
# Number of dataset classes.
num_classes = 4

# Training batch size and iteration limit.
batch_size = 64
max_batches = 10000

# Learning-rate schedule.
learning_rate = 0.00261
learning_rate_decay = 0.0005
learning_rate_decay_steps = '4800, 5400'

# Optimizer momentum.
momentum = 0.9

# Directory for training output.
output_dir = 'data/out'

# Select one of the commands below for training, evaluation, or testing.
# Training:
# command = f"darknet.exe detector train {data_file} {config_file} {pretrained_weights} -map"
# Mean average precision evaluation:
command = f"darknet.exe detector map {data_file} {config_file} backup/custom-yolov4-tiny-detector_4class_416_final.weights"
# Single-image test:
# command = f"darknet.exe detector test {data_file} {config_file} backup/custom-yolov4-tiny-detector_4class_final.weights -thresh 0.5"
# os.system(command)
print(command)
