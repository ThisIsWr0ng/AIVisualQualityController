import os

# Set up training file directories for custom dataset
os.chdir('C:/Darknet')
dataset = 'E:\Downloads\QC.v10i.darknet'
os.system(f"copy {dataset}\\train\\_darknet.labels data\\obj.names")
os.mkdir("C:\\Darknet\\data\\obj")

# Copy image and labels
os.system(f"copy {dataset}\\train\\*.jpg C:\\Darknet\\data\\obj\\")
os.system(f"copy {dataset}\\valid\\*.jpg C:\\Darknet\\data\\obj\\")
os.system(f"copy {dataset}\\train\\*.txt C:\\Darknet\\data\\obj\\")
os.system(f"copy {dataset}\\valid\\*.txt C:\\Darknet\\data\\obj\\")

with open("C:\\Darknet\\data\\obj.data", "w") as out:
    out.write("classes = 4\n")
    out.write("train = C:\\Darknet\\data\\train.txt\n")
    out.write("valid = C:\\Darknet\\data\\valid.txt\n")
    out.write("names = C:\\Darknet\\data\\obj.names\n")
    out.write("backup = backup\\")

# Write train file (just the image list)
with open("C:\\Darknet\\data\\train.txt", "w") as out:
    for img in [f for f in os.listdir(f"{dataset}\\train") if f.endswith("jpg")]:
        out.write(f"data/obj/{img}\n")

# Write the valid file (just the image list)
with open("C:\\Darknet\\data\\valid.txt", "w") as out:
    for img in [f for f in os.listdir(f"{dataset}\\valid") if f.endswith("jpg")]:
        out.write(f"C:\\Darknet\\data\\obj\\{img}\n")

#train
#os.system(f"cd C:\darknet")
#os.system(f"darknet.exe detector train data/obj.data cfg/custom-yolov4-tiny-detector.cfg yolov4-tiny.conv.29 -dont_show -map")
