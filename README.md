# Forest-Fire-Detection (ongoing)

The Forest Fire Detection system is designed to identify and localize fire hotspots in images using a combination of a classifier and an object detection model. This system integrates data collection, model training, and deployment strategies to ensure effective wildfire management.

## Table of Contents

1. [Overview](#overview)
2. [Data Collection](#data-collection)
3. [Model Selection and Training](#model-selection-and-training)
4. [Model Features](#model-features)

## Overview

The Forest Fire Detector consists of two main components:
- **Classifier:** A RESNET-50 model that filters images to identify those likely containing smoke or fire.
- **Object Detection Model:** A YOLOv5 or YOLOv8 model that locates fire and smoke within the filtered images.

## Data Collection

### Types of Data Required
- **Aerial Images:** Captured by drones or satellites showing forested areas with and without fire.
- **Ground-Level Images:** Photos or videos taken from ground surveillance cameras in forest areas.

### Data Sources
Analaysied, filtered and merged data from the following sources.
- **Kaggle:** [The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)
- **Roboflow:**
  - [Forest Fire Dataset (KZDLm)](https://universe.roboflow.com/aastmt-q0keu/forest-fire-kzdlm)
  - [Fire Detection Dataset (TIYPl)](https://universe.roboflow.com/ssanne/fire-tiypl/browse/fork?queryText=&pageSize=200&startingIndex=0&browseQuery=true)
  - [Forest Fire Preprocessing](https://app.roboflow.com/hydroponics-romfd/forest-fire-ke3xv-kid4l/generate/preprocessing)

### Data Annotation
- **Tools:** Use Roboflow for labeling and annotating fire regions in images.
- **Annotations:** Create bounding boxes around fire hotspots for training object detection models.

## Model Selection and Training

### Classifier
- **Model Type:** RESNET-50
- **Training Data:** Approximately 5,000 images
- **Purpose:** Filter images to pass only those with potential smoke or fire to the object detection model.

### Object Detection Model
- **Model Type:** YOLOv5 or YOLOv8
- **Purpose:** Detect and localize fire hotspots within images.

### Optional: Segmentation Model
- **Model Type:** U-Net
- **Purpose:** Provide detailed segmentation of fire regions at the pixel level.

### Libraries Required
- **TensorFlow and PyTorch:** For building, training, and fine-tuning models.
- **OpenCV:** For image preprocessing, data augmentation, and integrating vision systems.
- **Roboflow:** For dataset management and preprocessing.

### Training Process
1. **Train Classifier:**
   - Train the RESNET-50 model to filter out non-fire images.
2. **Train Object Detection Model:**
   - Train YOLOv5 or YOLOv8 to detect and localize fire hotspots using annotated data.


### Model Integration
- Use the classifier to filter images, then pass the filtered images to the object detection model for precise localization. If a segmentation model is used, it provides additional detail about fire regions.

## Model Features

- **Real-time Detection:** Detect fire hotspots in real-time from both aerial and ground-level images.
- **Scalability:** Handle various resolutions and image sources.
- **Multimodal Analysis (Optional):** Incorporate thermal data to enhance detection accuracy.
