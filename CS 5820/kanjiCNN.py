import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. Hyperparameters
# =============================
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 49
MODEL_PATH = "best_kanji_cnn_model.pth"
INPUT_SIZE = (112, 112)

# =============================
# 2. Device Detection
# =============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# =============================
# 3. Custom Dataset
# =============================
class Kuzushiji49NumpyDataset(Dataset):
    def __init__(self, imgs_path, labels_path, transform=None):
        imgs_npz = np.load(imgs_path)
        labels_npz = np.load(labels_path)
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
# 4. Transforms
# =============================
# Load your training images
imgs_npz = np.load("./data/hiragana_final/hiragana-train-imgs.npz")
images = imgs_npz[imgs_npz.files[0]]  # Assuming first key contains image data

# Compute mean and std across all pixels
mean = images.mean() / 255.0  # Normalize to [0,1] range if original is 0-255
std = images.std() / 255.0

print(f"{mean}, {std}")

transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)), 
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# =============================
# 5. Load Data
# =============================
train_dataset = Kuzushiji49NumpyDataset(
    "./data/hiragana_final/hiragana-train-imgs.npz",
    "./data/hiragana_final/hiragana-train-labels.npz",
    transform=transform
)

test_dataset = Kuzushiji49NumpyDataset(
    "./data/hiragana_final/hiragana-test-imgs.npz",
    "./data/hiragana_final/hiragana-test-labels.npz",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============================
# 6. CNN Model
# =============================
class KanjiCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(KanjiCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

model = KanjiCNN().to(device)

# =============================
# 7. Loss, Optimizer, Scheduler
# =============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# =============================
# 8. Training Loop with Metrics Tracking
# =============================
losses, train_accs, val_accs, balanced_accs = [], [], [], []
best_accuracy = 0.0

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    # Validation
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = 100 * correct / total

    # Save best model
    if val_accuracy > best_accuracy:
        best_accuracy = balanced_accuracy
        torch.save(model.state_dict(), MODEL_PATH)

    # Store metrics
    losses.append(avg_loss)
    train_accs.append(train_accuracy)
    val_accs.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

    scheduler.step()

end_time = time.time()
total_time = end_time - start_time
print(f"Total Training Time: {total_time/60:.2f} minutes")

# =============================
# 9. Compute Percentage Changes
# =============================
pct_loss = [0] + [((losses[i] - losses[i-1]) / losses[i-1]) * 100 for i in range(1, len(losses))]
pct_train = [0] + [((train_accs[i] - train_accs[i-1]) / train_accs[i-1]) * 100 for i in range(1, len(train_accs))]
pct_val = [0] + [((val_accs[i] - val_accs[i-1]) / val_accs[i-1]) * 100 for i in range(1, len(val_accs))]
pct_balanced = [0] + [((balanced_accs[i] - balanced_accs[i-1]) / balanced_accs[i-1]) * 100 for i in range(1, len(balanced_accs))]

# =============================
# 10. Plot Combined Figure
# =============================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Metrics per Epoch
ax1.plot(range(1, EPOCHS+1), losses, label='Loss', color='red')
ax1.plot(range(1, EPOCHS+1), train_accs, label='Train Acc', color='blue')
ax1.plot(range(1, EPOCHS+1), val_accs, label='Val Acc', color='green')
ax1.plot(range(1, EPOCHS+1), balanced_accs, label='Balanced Acc', color='purple')
ax1.set_title('Metrics per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Value')
ax1.legend()
ax1.grid(True)

# Percentage Change per Epoch
ax2.plot(range(1, EPOCHS+1), pct_loss, label='Loss % Change', color='red')
ax2.plot(range(1, EPOCHS+1), pct_train, label='Train Acc % Change', color='blue')
ax2.plot(range(1, EPOCHS+1), pct_val, label='Val Acc % Change', color='green')
ax2.plot(range(1, EPOCHS+1), pct_balanced, label='Balanced Acc % Change', color='purple')
ax2.set_title('Percentage Change per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('% Change')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_metrics_combined.png')
print("Combined figure saved as training_metrics_combined.png")
