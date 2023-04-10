import subprocess

# Define the path to your virtual environment
venv_path = "/home/dawid/Git/AIVisualQualityController/venv_tflite"

# Activate the virtual environment
activate_cmd = f"source {venv_path}/bin/activate"
subprocess.run(activate_cmd, shell=True, executable="/bin/bash")

# Run your webcam detection script (replace 'your_detection_script.py' with your script's name)
webcam_detection_cmd = "Webcam_MobileNet_v3.py"
subprocess.run(webcam_detection_cmd, shell=True, executable="/bin/bash")

# Deactivate the virtual environment
deactivate_cmd = "deactivate"
subprocess.run(deactivate_cmd, shell=True, executable="/bin/bash")