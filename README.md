# Object Classification with YOLOv8

## Overview

This project uses **YOLOv8**, a deep learning model, for real-time object classification in images. It detects objects, draws color-coded bounding boxes around them, and displays their confidence scores. The system leverages **convolutional neural networks (CNNs)** and **non-maximum suppression (NMS)** for accurate predictions.

## Features
- **Object detection**: Identifies multiple objects within an image and draws bounding boxes around them.
- **Color-coded bounding boxes**: Each object is assigned a unique color based on its label.
- **Confidence scores**: Each bounding box is annotated with the model's confidence score (probability of correct detection).
- **Non-maximum suppression**: Removes redundant overlapping bounding boxes by evaluating their intersection over union (IoU).

## Technical Details

### 1. **Model Used**
The model is built using **YOLOv8**, which is part of the **YOLO (You Only Look Once)** series of models. YOLOv8 is a state-of-the-art, fast object detection model that predicts both **bounding box coordinates** and **class labels** in one forward pass of the network.

### 2. **Algorithm**

- **Grid-based prediction**: The image is divided into a grid. Each grid cell predicts bounding boxes and class probabilities.
- **Bounding Box Regression**: The model directly predicts bounding box coordinates (x1, y1, x2, y2) along with the **confidence score**, which measures the likelihood that an object exists within the predicted box.
- **Non-maximum Suppression (NMS)**: Used to reduce multiple overlapping bounding boxes. NMS filters out boxes based on their **Intersection over Union (IoU)**, keeping the box with the highest confidence score.
  
### 3. **Mathematical and CS Details**

- **Confidence Scores**: The confidence score is predicted using **logistic regression** and indicates the probability that an object is within the predicted bounding box.
- **IoU (Intersection over Union)**: This metric is used to measure the overlap between two bounding boxes. If the IoU of two boxes exceeds a predefined threshold, the one with the lower confidence score is discarded.
- **Logistic Regression**: The model uses logistic regression to predict probabilities of object presence in grid cells.
- **CNN Architecture**: The YOLO model uses **Convolutional Neural Networks (CNNs)** for feature extraction and classification. The CNN layers detect complex patterns and high-level features in the image.

### 4. **Post-Processing**
After detection, the system applies **random RGB color coding** for the bounding boxes and labels using a simple hashing function. This makes each class label visually distinct and helps differentiate between objects.

### 5. **Optimizations**
The bounding box thickness has been increased to **4px** for better visibility, and a background rectangle is added behind each label to ensure text visibility on complex images.

