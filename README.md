# HydroVision

Welcome to HydroVision! This is an underwater object detection project designed to detect various objects in underwater imagery. It leverages lightweight deep learning models that mimic the algorithms of YOLO (You Only Look Once) and Vision Transformer (ViT) architectures, with a focus on achieving decent accuracy using computationally efficient approaches. The project compares three models: a lightweight ViT mimic, a lightweight YOLO mimic, and a combined architecture integrating both approaches.

**Important Note:** I ran it for just 1 epoch to check if the model runs correctly or not. Whoever wishes to use the model for good enough results, may run it for 20-25 epochs on his own CPU and 40-50 epochs using a High Performance Computing (HPC) server or GPU.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

HydroVision aims to provide a computationally efficient approach for underwater object detection. The project focuses on lightweight model architectures that strike a balance between accuracy and efficiency, enabling the detection of objects such as fish, wrecks, aquatic plants, human divers, robots, and the sea floor with minimal computational resources.

## Dataset

The dataset used for this project is USIS10K, which consists of underwater images categorized into multiple classes as:

```
class_names = ["wrecks/ruins", "fish", "reefs", "aquatic plants", "human divers", "robots", "sea-floor"].
```

The dataset is structured as follows:

```
data
  ├── USIS10K
  │   ├── multi_class_annotations
  │   │   ├── multi_class_train_annotations.json
  │   │   ├── multi_class_val_annotations.json
  │   │   ├── multi_class_test_annotations.json
  │   ├── train
  │   │   ├── train_00001.jpg
  │   │   ├── ...
  │   ├── train_labels
  │   │   ├── train_00001.txt
  │   │   ├── ...
  │   ├── val
  │   │   ├── val_00001.jpg
  │   │   ├── ...
  │   ├── val_labels
  │   │   ├── val_00001.txt
  │   │   ├── ...
  │   ├── test
  │   │   ├── test_00001.jpg
  │   │   ├── ...
  │   ├── test_labels
  │   │   ├── test_00001.txt
  │   │   ├── ...
```
The annotations are parsed into text files, with each line representing an object in the image in the format:
```
<class_id> <x_center> <y_center> <width> <height>
```

## Model Architectures

Three models are implemented in this project:

1. **Lightweight ViT Mimic**: A simplified Vision Transformer model with fewer transformer blocks and attention heads.
2. **Lightweight YOLO Mimic**: A basic convolutional model inspired by YOLO, with downsampling and prediction heads for bounding boxes and class labels.
3. **Integrated Model (LightViT-YOLO)**: Combines the outputs of the ViT and YOLO mimics for improved performance.

## Installation

To set up this project, follow these steps:

1. **Clone the repository:**
   ``` 
   git clone https://github.com/rujuldwivedi/Underwater-Object-Detection.git
   cd Underwater-Object-Detection
   ```

2. **Install the required packages:**
   ``` 
   pip install -r requirements.txt
   ```

3. **Organize the dataset as specified in the [Dataset](#dataset) section.**

## Usage

1. **Training the models:**
   - Train each model separately using the training script provided:
     ``` 
     python train_vit.py  # For Lightweight ViT
     python train_yolo.py  # For Lightweight YOLO
     python train_integrated.py  # For the Integrated Model
     ```

2. **Evaluating the models:**
   - Evaluate the models on the test set:
     ``` 
     python evaluate.py --model vit
     python evaluate.py --model yolo
     python evaluate.py --model integrated
     ```

3. **Inference on new images:**
   - Run inference on a set of images:
     ``` 
     python infer.py --image_path <sample-image-path>.jpg --model integrated
     ```

## Results

The project compares the performance of the three models based on accuracy, inference time, and computational requirements. Detailed results can be added after completing the evaluations.

## Acknowledgements

This project would not have been possible without the guidance and support of [**Dr. Shitala Prasad**](https://www.linkedin.com/in/shitalaprasad/), whose expertise and insights were invaluable throughout the development process.

