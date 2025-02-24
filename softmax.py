import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from PIL import Image
from datetime import datetime

output_file = 'Softmax_results.txt'
dataset_path = './removed_bg_dataset'

# Parameters
image_size = (128, 128)
num_classes = 10
batch_size = 32
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images and labels
data, labels = [], []
for entry in os.scandir(dataset_path):
    if entry.is_dir():
        for img_name in os.listdir(entry.path):
            img_path = os.path.join(entry.path, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("L").resize(image_size)  # Grayscale & Resize
                    data.append(np.array(img).flatten())  # Flatten image
                    labels.append(entry.name)
            except Exception:
                print(f"Skipping: {img_path}")

# Convert data to tensors
data = torch.tensor(np.array(data, dtype=np.float32) / 255.0)
labels = torch.tensor(LabelEncoder().fit_transform(labels), dtype=torch.long)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# PyTorch Dataset and DataLoader
class ImageDataset(Dataset):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

train_loader = DataLoader(ImageDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(ImageDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Model Definition
class SoftmaxModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x): return self.model(x)

model = SoftmaxModel(X_train.shape[1], num_classes).to(device)

# Training
criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
correct, total, all_preds, all_labels = 0, 0, [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        preds = outputs.argmax(1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# Results
test_acc = correct / total
f1 = f1_score(all_labels, all_preds, average='weighted')
results = f"Test Accuracy: {test_acc:.4f}, F1 Score: {f1:.4f}, Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {image_size}"
print(results)

with open(output_file, 'a') as f:
    f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{results}\n\n")
