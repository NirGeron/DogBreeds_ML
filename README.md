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
- **Source:** [Kaggle - Doges 77 Breeds](https://www.kaggle.com/datasets)
- The dataset contains labeled images of 77 different dog breeds.
- Preprocessing includes resizing images, normalization, and data augmentation.

## Installation
Ensure you have Python installed (preferably 3.8+), then install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Train a Model
To train a specific model, run:
```bash
python train.py --model svm  # Options: 'svm', 'boosting', 'softmax'
```

### 2. Evaluate Model
Run evaluation on test data:
```bash
python evaluate.py --model svm
```

### 3. Predict a Dog Breed
Use a trained model to predict a breed from an image:
```bash
python predict.py --image path/to/image.jpg --model svm
```

## Results
- Model performance is evaluated using **accuracy, precision, recall, and F1-score**.
- **Comparison of different models** to determine the best classifier.

## Future Improvements
- Implement deep learning models (e.g., CNNs) for better accuracy.
- Optimize hyperparameters for boosting models.
- Deploy as a web application for easy usage.

## Contributors
- **Nir Geron** - [GitHub](https://github.com/NirGeron)
- **Dor Shir** - [GitHub](https://github.com/Dorshir)
