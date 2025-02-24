import os
import numpy as np
import pandas as pd
from PIL import Image

from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score

import matplotlib.pyplot as plt
from datetime import datetime

def load_images_from_folders(data_dir, image_size=(64, 64), limit=None, save_path=None, load_cached=False):
    """
    Loads images from subdirectories where each subdirectory represents a class.
    Saves the processed dataset to a file and loads it from cache if available.
    """
    if load_cached and save_path and os.path.exists(save_path):
        print(f"Loading dataset from cache: {save_path}")
        data = np.load(save_path, allow_pickle=True)
        return data["X"], data["y"]

    all_images = []
    all_labels = []
    class_names = sorted(os.listdir(data_dir))  # Get class folder names
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # Skip files if any
        
        image_files = os.listdir(class_dir)
        if limit is not None:
            image_files = image_files[:limit]  # Apply limit per class
        
        for image_file in image_files:
            img_path = os.path.join(class_dir, image_file)
            try:
                with Image.open(img_path) as pil_img:
                    pil_img = pil_img.convert('RGB')
                    pil_img = pil_img.resize(image_size, Image.BILINEAR)
                    img_np = np.array(pil_img)
                    all_images.append(img_np)
                    all_labels.append(class_name)  # Use folder name as label
            except Exception as e:
                print(f"Skipping file {img_path}: {e}")
    
    X = np.array(all_images)
    y = np.array(all_labels)

    # Save the dataset if a save_path is provided
    if save_path:
        np.savez_compressed(save_path, X=X, y=y)
        print(f"Dataset saved to {save_path}")

    return X, y

# Set paths and parameters
data_dir = './dataset'  # The directory containing class subfolders
IMAGE_SIZE = (64, 64)
CACHE_PATH = "dataset_cache.npz"  # Path to save/load dataset

# Load and save data
X, y = load_images_from_folders(data_dir=data_dir, image_size=IMAGE_SIZE, save_path=CACHE_PATH, load_cached=True)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

# Normalize pixel values to [0, 1]
X = X / 255.0
# First split into train + (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=42)

# Then split (validation + test) into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) 
from sklearn.ensemble import AdaBoostClassifier

if len(X_train.shape) > 2:
    X_train = X_train.reshape(X_train.shape[0], -1)

# Ensure y_train is 1D
if len(y_train.shape) > 1:
    y_train = y_train.ravel()    

estimators = 50

# Initialize AdaBoost
model = AdaBoostClassifier(
    n_estimators=estimators
)

# Train the model
model.fit(X_train, y_train)

if len(X_test.shape) > 2:
    X_test = X_test.reshape(X_test.shape[0], -1)

# Make Predictions (on train and test sets)
y_pred_test = model.predict(X_test)     # Predictions on test set
y_pred_train = model.predict(X_train)  # Predictions on train set

# Compute evaluation metrics
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, average='weighted')
report = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, zero_division=0)

print(f'Training Accuracy: {accuracy_train:.3f}')
print(f'Test Accuracy: {accuracy_test:.3f}')
print(f'F1 Score: {f1_test:.3f}')
print("Classification Report:")
print(report)

# Log current evaluation details
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
details = (
    f"Training Accuracy: {accuracy_train:.3f}\n"
    f"Test Accuracy: {accuracy_test:.3f}\n"
    f"F1 Score: {f1_test:.3f}\n"
    f"Image Size: {IMAGE_SIZE}\n"
    f"Num of Estimators: {estimators}\n"
)

output_file = "Adaboost_results.txt"

# Save the results to the output file
with open(output_file, 'a') as f:
    f.write(f"Report generated at: {current_time}\n")
    f.write(details + "\n")
    f.write(report + "\n\n")
