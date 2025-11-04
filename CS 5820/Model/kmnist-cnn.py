# Contains code for the machine model 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# =============================
# 1. Hyperparameters
# =============================
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 10  # Kuzushiji-MNIST has 10 classes
MODEL_PATH = "kanji_cnn_model.pth"


# =============================
# 2. Data Loading & Preprocessing
# =============================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download Kuzushiji-MNIST dataset
# Formatting the image and labels into NumPy arrays

X_train = np.load('kmnist-train-imgs.npz')['arr_0']
y_train = np.load('kmnist-train-labels.npz')['arr_0']

X_test = np.load('kmnist-test-imgs.npz')['arr_0']
y_test = np.load('kmnist-test-labels.npz')['arr_0']

# 1. Convert NumPy arrays to Pytorch tensors
X_train_tensor = torch.from_numpy(X_train).float() / 255.0  #Normalize and convert type
y_train_tensor = torch.from_numpy(y_train).long()

# 2. Create a TensorDataset from tensors
train_dataset= TensorDataset(X_train_tensor, y_train_tensor)

# 3. Create the DataLoader for batching and shuffling
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)
# Now we can iterate over train_loader in the training loop



# ==== We might not need this because we are creating a Dataset from tensors.
# class KanjiCNN(nn.Module):
#     def __init__(self):
#         super(KanjiCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 14 * 14, 128)
#         self.fc2 = nn.Linear(128, NUM_CLASSES)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.25)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 14 * 14)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# model = KanjiCNN()
# =============================



# =============================
# 4. Loss & Optimizer
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================
# 5. Training Loop
# =============================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# =============================
# 6. Validation
# =============================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

# =============================
# 7. Save Model
# =============================
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
 