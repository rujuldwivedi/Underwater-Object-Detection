# HydroVision

## Overview
Welcome to **HydorVision**! This repository explores the task of detecting underwater objects using deep learning models. It includes implementations using various approaches and pretrained models, focusing on fish detection and plastic mugs detection among other classes.

## Datasets
Two main datasets were used for training and evaluation:
- Dataset 1: [Kaggle Dataset Name 1](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots)
- Dataset 2: [Kaggle Dataset Name 2](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)

## Models Used
### Transfer Learning Models:
1. **MobileNetV2**:
   - Details: Pretrained on ImageNet, fine-tuned for underwater object detection.
   
2. **YOLOv5**:
   - Details: Implemented using YOLOv5 repository with modifications for underwater classes.
   - Cloned Repo: [YOLOv5 Repository](https://github.com/ultralytics/yolov5)

### Basic Model:
- A simple baseline model was also implemented to compare transfer learning performance.

## Usage
### Training
1. **MobileNetV2**: 
python train_mobilenet.py --dataset path_to_dataset --epochs num_epochs

2. **YOLOv5**: 
python train_yolov5.py --data path_to_data.yaml --cfg path_to_config.yaml --epochs num_epochs

### Evaluation
Run evaluation scripts to assess model performance on test datasets.

### Inference
Use trained models for inference on new images or video streams.

## Contributions
Contributions and improvements are welcome. Please fork the repository and submit pull requests for review.

## Acknowledgements

I would like to sincerely thank Dr. Shitala Prasad for his expert guidance and unwavering support throughout the "Underwater Object Detection" project. His deep knowledge and innovative ideas greatly enriched this work, and his mentorship has been pivotal in bringing this project to fruition.
