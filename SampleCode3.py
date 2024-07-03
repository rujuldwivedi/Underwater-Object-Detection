import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Define directories
IMAGE_DIR = 'dataset/images'
LABEL_DIR = 'dataset/labels'
IMAGE_SIZE = (224, 224)

# Function to load data
def load_data(image_dir, label_dir):
    paths = []
    labels = []
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    for file in label_files:
        image_path = os.path.join(image_dir, file.replace('.txt', '.jpg'))
        label_path = os.path.join(label_dir, file)
        
        if os.path.exists(image_path):
            with open(label_path, 'r') as f:
                label_info = f.readline().strip().split()
                class_index = int(label_info[0])
                bbox_x = float(label_info[1])
                bbox_y = float(label_info[2])
                bbox_width = float(label_info[3])
                bbox_height = float(label_info[4])
                
            paths.append(image_path)
            labels.append([class_index, bbox_x, bbox_y, bbox_width, bbox_height])
    
    return pd.DataFrame({'path': paths, 'label': labels})

# Load data
train_df = load_data(os.path.join(IMAGE_DIR, 'train'), os.path.join(LABEL_DIR, 'train'))
valid_df = load_data(os.path.join(IMAGE_DIR, 'valid'), os.path.join(LABEL_DIR, 'valid'))
test_df = load_data(os.path.join(IMAGE_DIR, 'test'), os.path.join(LABEL_DIR, 'test'))

# Function to parse YOLO labels
def parse_yolo_labels(label):
    class_index = label[0]
    bbox_x = label[1]
    bbox_y = label[2]
    bbox_width = label[3]
    bbox_height = label[4]
    
    # You can further process this information as per your model's requirements
    return class_index, bbox_x, bbox_y, bbox_width, bbox_height

# Define data generators (you can use ImageDataGenerator as well)
def data_generator(df, batch_size=32):
    while True:
        df = shuffle(df)
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_paths = df['path'][start:end]
            batch_labels = df['label'][start:end]
            
            batch_images = []
            batch_parsed_labels = []
            
            for i, path in enumerate(batch_paths):
                image = tf.keras.preprocessing.image.load_img(path, target_size=IMAGE_SIZE)
                image = tf.keras.preprocessing.image.img_to_array(image)
                image = image / 255.0  # Normalize to [0, 1]
                batch_images.append(image)
                
                parsed_label = parse_yolo_labels(batch_labels.iloc[i])
                batch_parsed_labels.append(parsed_label)
            
            yield np.array(batch_images), np.array(batch_parsed_labels)

# Create data iterators
train_generator = data_generator(train_df)
valid_generator = data_generator(valid_df)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5)  # Adjust output size as per your label parsing
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // 32,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=len(valid_df) // 32
)

# Evaluate the model
test_images, test_labels = next(data_generator(test_df, len(test_df)))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Make predictions
predictions = model.predict(test_images)