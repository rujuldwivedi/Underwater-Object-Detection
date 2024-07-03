import os
import subprocess

# Update these paths with your dataset directory and YAML file location
dataset_path = 'dataset'  # Replace with your dataset path
yaml_path = 'underwater.yaml'  # Replace with your YAML path

# Load the test image
image_path = os.path.join(dataset_path, 'test', 'images', 'IMG_2289_jpeg_jpg.rf.fe2a7a149e7b11f2313f5a7b30386e85.jpg')

# Build command to execute detect.py
command = f"python yolov5/detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source {image_path}"

# Execute command and capture stdout
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Print the captured output
print(result.stdout)