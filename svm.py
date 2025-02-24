import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
from datetime import datetime

output_file = 'SVM_results.txt'
dataset_path = './removed_bg_dataset'
image_size = (128, 128)
num_pca_components = 150  # Number of PCA components to retain

# Lists to store data
data = []
labels = []

# Load images from dataset
for breed in os.listdir(dataset_path):
    breed_path = os.path.join(dataset_path, breed)
    if os.path.isdir(breed_path):
        for img_name in os.listdir(breed_path):
            img_path = os.path.join(breed_path, img_name)
            try:
                # Ensure the file is a valid image
                with Image.open(img_path) as img:
                    img.verify()  # Check integrity
                
                # Read and preprocess the image
                image = cv2.imread(img_path)
                image = cv2.resize(image, image_size)  # Resize the image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale (reduces dimensions)
                data.append(image.flatten())  # Flatten the image into a vector
                labels.append(breed)
            except Exception as e:
                print(f"Skipping invalid image: {img_path}")

# Convert lists to NumPy arrays
data = np.array(data, dtype="float32") / 255.0  # Normalize pixel values
labels = np.array(labels)

# Encode class labels as integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Reduce dimensions using PCA
pca = PCA(n_components=num_pca_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train an SVM model
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_model.fit(X_train_pca, y_train)

# Make predictions and evaluate performance
y_pred = svm_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
details = f"Test Accuracy: {accuracy:.4f} image_size: {image_size} PCA: {num_pca_components}"
output = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(current_time)
print(details)
print(output)

with open(output_file, 'a') as f:
    f.write(f"Report generated at: {current_time}\n for {dataset_path} \n {details}\n {output} \n\n")
