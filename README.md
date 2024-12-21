# **Deep Learning for Pose Classification and Depth Estimation in Microrobotics**

This project explores the implementation of deep learning models for **Pose Classification** and **Depth Estimation** in microrobotics. Leveraging techniques such as **data preprocessing**, **transfer learning**, and **hyperparameter tuning**, this study evaluates the effectiveness of different model architectures on a grayscale microrobot image dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Preparation](#dataset-preparation)
- [Tasks Overview](#tasks-overview)
  - [Task 1: Data Preparation and Preprocessing](#task-1-data-preparation-and-preprocessing)
  - [Task 2: Pose Classification](#task-2-pose-classification)
  - [Task 3: Depth Estimation](#task-3-depth-estimation)
  - [Task 4: Hyperparameter Tuning](#task-4-hyperparameter-tuning)
- [Results](#results)
- [Requirements](#requirements)

---

## Project Overview

Microrobotics relies on precise **pose estimation** and **depth perception** for navigation and control. This project implements and compares **Simple CNN**, **Deeper CNN**, and **MobileNetV2** architectures for the following tasks:
1. **Pose Classification**: Predicting discrete pose classes.
2. **Depth Estimation**: Estimating continuous depth values.

Key contributions include:
- Data preprocessing, including **augmentation**, **normalization**, and **stratified splits**.
- Implementation of **transfer learning** with MobileNetV2.
- Hyperparameter tuning using **Bayesian Optimization**.

---

## Dataset Preparation

- **Data Source**: Grayscale images of microrobots with labeled poses and depths.
- **Size**: 2,016 images with significant class imbalance in pose labels.
- **Preprocessing**:
  - Data augmentation: Rotation, translation, zoom, and horizontal flips.
  - Depth normalization: Zero-mean and unit-variance scaling.
  - Stratified train-validation-test splits.

---

## Tasks Overview

### **Task 1: Data Preparation and Preprocessing**
- **Objective**: Load, preprocess, and analyze the dataset.
- **Steps**:
  1. Extract the dataset and load labels.
  2. Analyze distributions (Pose Classes and Depth Values).
  3. Apply data augmentation.
  4. Normalize pixel values and standardize depth.
  5. Split the dataset into **train**, **validation**, and **test** sets.

[Sample Code for Task 1](https://colab.research.google.com/drive/1ZOar1-MAWiCQtzJ4UT9Pl9vfdO-vWDwL?usp=sharing)

---

### **Task 2: Pose Classification**
- **Objective**: Classify poses using three CNN architectures:
  1. **Simple CNN**: Baseline model with minimal complexity.
  2. **Deeper CNN**: Enhanced model with additional layers for feature extraction.
  3. **MobileNetV2**: Transfer learning model with pre-trained ImageNet weights.
- **Key Techniques**:
  - Data augmentation with `ImageDataGenerator`.
  - Categorical cross-entropy loss for multi-class classification.
  - Metrics: Accuracy, Precision, Recall, F1-Score.

---

### **Task 3: Depth Estimation**
- **Objective**: Predict depth values as a regression task.
- **Steps**:
  1. Adapt CNN models for regression with a linear output layer.
  2. Train and evaluate models using RMSE as the primary metric.
  3. Perform error analysis with scatter plots and residual distributions.

---

### **Task 4: Hyperparameter Tuning**
- **Objective**: Optimize MobileNetV2 hyperparameters for depth estimation.
- **Key Hyperparameters**:
  - Unfrozen layers: {0, 20, 50}.
  - Dense units: {64, 128, 256}.
  - Dropout rates: {0.3, 0.5}.
  - Learning rates: {1e-4, 1e-5}.
- **Tuning Framework**: Bayesian Optimization using `Keras Tuner`.

---

## Results

### Pose Classification
- **Best Model**: MobileNetV2
  - **Accuracy**: 60.89%.
  - **Misclassifications**: Often between visually similar poses.

### Depth Estimation
- **Best Model**: MobileNetV2
  - **RMSE**: 0.1652.
  - **Error Analysis**: Tight residual distributions and high alignment in scatter plots.

### Hyperparameter Tuning
- Fine-tuning MobileNetV2 increased RMSE due to overfitting, highlighting challenges with small datasets.

---

## Requirements
- Python 3.8+
- TensorFlow/Keras
- Keras Tuner
- NumPy, Matplotlib, Seaborn, scikit-learn

### Run the Colab notebook: 
https://colab.research.google.com/drive/1ZOar1-MAWiCQtzJ4UT9Pl9vfdO-vWDwL?usp=sharing


