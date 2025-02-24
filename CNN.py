import os
import copy
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
                    # pil_img = pil_img.resize(image_size, Image.BILINEAR)
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
IMAGE_SIZE = (128, 128)
CACHE_PATH = "dataset_cache.npz"  # Path to save/load dataset

# Load and save data
X, y = load_images_from_folders(data_dir=data_dir, image_size=IMAGE_SIZE, save_path=CACHE_PATH, load_cached=True)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
# First split into train + (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=42)

# Then split (validation + test) into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) 
# Calculate mean and standard deviation from the training group
train_mean = X_train.mean(axis=(0, 1, 2))
train_std = X_train.std(axis=(0, 1, 2))

# Normalize the training data
X_train_normalized = (X_train - train_mean) / train_std

# Apply the same normalization to the validation and test datasets
X_val_normalized = (X_val - train_mean) / train_std
X_test_normalized = (X_test - train_mean) / train_std

# Transpose axes for PyTorch
X_train = np.transpose(X_train_normalized, (0, 3, 1, 2))
X_val = np.transpose(X_val_normalized, (0, 3, 1, 2))
X_test = np.transpose(X_test_normalized, (0, 3, 1, 2))

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class ImageDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ImageDataset(X_train_tensor, y_train_tensor)
val_dataset = ImageDataset(X_val_tensor, y_val_tensor)
test_dataset = ImageDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class DogBreedCNN(nn.Module):
    
    def __init__(self, num_classes):
        
        super(DogBreedCNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv1 = nn.Dropout(p=0.2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout_conv2 = nn.Dropout(p=0.2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout_conv3 = nn.Dropout(p=0.2)

        # Fully Connected Layers
        # self.fc1 = nn.Linear(128 * 8 * 8, 256) # (64, 64)
        self.fc1 = nn.Linear(128 * 16 * 16, 256) # (128, 128)
        # self.fc1 = nn.Linear(128 * 28 * 28, 256) # (224, 224)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv1(x)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv2(x)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

num_classes = len(np.unique(y)) # Number of output classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DogBreedCNN(num_classes).to(device)

def train(model, train_loader, val_loader, num_epochs=50, learning_rate=0.01, patience=8, alpha=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=3, verbose=True
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')
    best_model_state = None
    no_improve_count = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_ce = criterion(outputs, labels)

            # L1 regularization
            l1_penalty = torch.tensor(0., requires_grad=True).to(device)
            for name, param in model.named_parameters():
                if "weight" in name and "bn" not in name:  # Exclude biases and BatchNorm parameters
                    l1_penalty += torch.norm(param, p=1)

            # Total loss
            total_loss = loss_ce + alpha * l1_penalty

            total_loss.backward()
            
            optimizer.step()
            
            running_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100. * correct / total)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss_ce = criterion(outputs, labels)

                # L1 regularization in validation
                l1_penalty = torch.tensor(0., requires_grad=False).to(device)
                for name, param in model.named_parameters():
                    if "weight" in name and "bn" not in name:  # Exclude biases and BatchNorm parameters
                        l1_penalty += torch.norm(param, p=1)

                total_loss = loss_ce + alpha * l1_penalty
                val_loss += total_loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100. * correct / total)
        
        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%")

        # Early stopping mechanism
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Restore the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses, train_accs, val_accs

num_of_epochs = 100
lr = 0.0005
train_losses, val_losses, train_accs, val_accs = train(model, train_loader, val_loader, num_epochs=num_of_epochs, learning_rate=lr)
