import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
from datetime import datetime

output_file = 'SVM_results.txt'
dataset_path = './removed_bg_dataset'
image_size = (128, 128)
num_pca_components = 150

# Load images and labels
data, labels = [], []
for entry in os.scandir(dataset_path):
    if entry.is_dir():
        for img_name in os.listdir(entry.path):
            img_path = os.path.join(entry.path, img_name)
            try:
                img = Image.open(img_path).convert("L").resize(image_size)  # Grayscale & resize
                data.append(np.array(img).flatten())  # Flatten image
                labels.append(entry.name)
            except:
                print(f"Skipping: {img_path}")

# Convert to NumPy arrays
data = np.array(data, dtype="float32") / 255.0  # Normalize
labels = LabelEncoder().fit_transform(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Apply PCA
pca = PCA(n_components=num_pca_components)
X_train_pca, X_test_pca = pca.fit_transform(X_train), pca.transform(X_test)

# Train & evaluate SVM
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X_train_pca, y_train)
y_pred = svm_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

# Save results
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
output = classification_report(y_test, y_pred, target_names=LabelEncoder().fit(labels).classes_)
result = f"Test Accuracy: {accuracy:.4f}, Image Size: {image_size}, PCA: {num_pca_components}"

print(current_time, result, output, sep="\n")

with open(output_file, 'a') as f:
    f.write(f"Report generated at: {current_time}\n{result}\n{output}\n\n")
