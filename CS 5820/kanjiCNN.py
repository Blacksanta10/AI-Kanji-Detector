import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# =============================
# 1. Hyperparameters
# =============================
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 49
MODEL_PATH = "kanji_cnn_model.pth"

# =============================
# 2. Custom Dataset
# =============================
class Kuzushiji49NumpyDataset(Dataset):
    def __init__(self, imgs_path, labels_path, transform=None):
        imgs_npz = np.load(imgs_path)
        labels_npz = np.load(labels_path)
        # Detect keys dynamically
        self.images = imgs_npz[imgs_npz.files[0]]
        self.labels = labels_npz[labels_npz.files[0]]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label

# =============================
# 3. Transforms
# =============================
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =============================
# 4. Load Data
# =============================
train_dataset = Kuzushiji49NumpyDataset(
    "./data/K49/k49-train-imgs.npz",
    "./data/K49/k49-train-labels.npz",
    transform=transform
)

test_dataset = Kuzushiji49NumpyDataset(
    "./data/K49/k49-test-imgs.npz",
    "./data/K49/k49-test-labels.npz",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============================
# 5. Define CNN Model
# =============================
class KanjiCNN(nn.Module):
    def __init__(self):
        super(KanjiCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = KanjiCNN()

# =============================
# 6. Loss & Optimizer
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================
# 7. Training Loop
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
# 8. Validation
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
# 9. Save Model
# =============================
torch.save(model.state_dict(), MODEL_PATH)

