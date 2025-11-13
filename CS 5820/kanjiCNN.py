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
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 46
MODEL_PATH = "best_kanji_cnn_model.pth"
INPUT_SIZE = (28, 28)

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
class Kuzushiji46NumpyDataset(Dataset):
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
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])


test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])



# =============================
# 5. Load Data
# =============================
train_dataset = Kuzushiji46NumpyDataset(
    "./data/hiragana_final/hiragana-train-imgs.npz",
    "./data/hiragana_final/hiragana-train-labels.npz",
    transform=train_transform
)

test_dataset = Kuzushiji46NumpyDataset(
    "./data/hiragana_final/hiragana-test-imgs.npz",
    "./data/hiragana_final/hiragana-test-labels.npz",
    transform=test_transform
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
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = KanjiCNN().to(device)

# =============================
# 7. Loss, Optimizer, Scheduler
# =============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
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

print(f"Loss {pct_loss}\n Train {pct_train}\n Val{pct_val}\n Balanced {pct_balanced}")

# =============================
# 10. Plot Separate Figures
# =============================
epochs = range(1, len(losses) + 1)

# --- Figure 1: Accuracies ---
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accs, label='Train Accuracy', color='royalblue', linewidth=2)
plt.plot(epochs, val_accs, label='Val Accuracy', color='seagreen', linewidth=2)
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)  # Ensure y-axis goes up to 100
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('accuracy_per_epoch.png', dpi=150)
plt.close()

# --- Figure 2: Loss ---
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, label='Training Loss', color='crimson', linewidth=2)
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.ylim(bottom=0)  # Ensure y-axis starts at 0
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('loss_per_epoch.png', dpi=150)
plt.close()

torch.save(model.state_dict(), MODEL_PATH)
print("✅ Model saved as best_kanji_cnn.pth")

