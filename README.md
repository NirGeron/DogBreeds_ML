# Dog Breeds Classification using Machine Learning
<p align="center">
  <img src="https://github.com/user-attachments/assets/9aa6ca8c-ffd5-42f4-b765-33dc5839729d"align="right">
</p>

## Overview
This project focuses on classifying dog breeds using machine-learning techniques. It utilizes the **Doges 10 Breeds** dataset from Kaggle and implements multiple classification models, including:

## Tools and Models
- **Support Vector Machine (SVM)**
- **Softmax Regression**
- **Convolutional Neural Network (CNN)**
- **Boosting models**

The goal is to compare the performance of these models in accurately classifying dog breeds based on image data.

## Dataset
- **Source:** [Kaggle - Doges 77 Breeds](https://www.kaggle.com/datasets/madibokishev/doges-77-breeds)
- The dataset contains labeled images of 77 different dog breeds.
- Preprocessing includes resizing images, normalization, and data augmentation.


## Usage
### 1. Download dataset
Download dataset from 
```
https://www.kaggle.com/datasets/madibokishev/doges-77-breeds
```

### 2. Unzip the dataset Remove the Background
run:
```bash
python unzip_dataset.py
python remove_bg.py
```

### 2. Train a Model
Choose a model to train


### 3. Predict a Dog Breed
Use a trained model to predict a breed from an image:


## Results
- Model performance is evaluated using **accuracy, precision, recall, and F1-score**.
- **Comparison of different models** to determine the best classifier.


## Contributors
- [Dor Shir](https://github.com/Dorshir)
- [Nir Geron](https://github.com/NirGeron)
