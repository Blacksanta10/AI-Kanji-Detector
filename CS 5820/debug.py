import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import pad
import threading
import time

labels = [
    "あ", "い", "う", "え", "お",
    "か", "き", "く", "け", "こ",
    "さ", "し", "す", "せ", "そ",
    "た", "ち", "つ", "て", "と",
    "な", "に", "ぬ", "ね", "の",
    "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も",
    "や", "ゆ", "よ",
    "ら", "り", "る", "れ", "ろ",
    "わ", "ゐ", "ゑ", "を",
    "ん", ""
]
# Force PyTorch single-thread mode
torch.set_num_threads(1)

# =============================
# Model Setup
# =============================
class KanjiCNN(nn.Module):
    def __init__(self):
        super(KanjiCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 49)
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

class ResizeWithPadding:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img.thumbnail((self.size, self.size), Image.LANCZOS)
        delta_w = self.size - img.width
        delta_h = self.size - img.height
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        return pad(img, padding, fill=0)

transform = transforms.Compose([
    transforms.Grayscale(),
    ResizeWithPadding(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cpu")
model = KanjiCNN().to(device)
model.load_state_dict(torch.load("kanji_cnn_model.pth", map_location=device))
model.eval()

# =============================
# Shared State
# =============================
latest_frame = None
prediction_text = "Loading..."
lock = threading.Lock()
running = True

# =============================
# Inference Thread
# =============================
def inference_loop():
    global latest_frame, prediction_text, running
    while running:
        if latest_frame is not None:
            with lock:
                frame_copy = latest_frame.copy()
            gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_frame, (28, 28))
            cv2.imwrite("test_image.jpg", resized_image)
            pil_frame = Image.fromarray(gray_frame)
            input_tensor = transform(pil_frame).unsqueeze(0).to(device)
            

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                idx = torch.argmax(probs).item()
                kanji_char = labels[idx]
                confidence = probs[idx].item() * 100

            prediction_text = f"{kanji_char} ({confidence:.1f}%)"
            print(f"[Prediction] {kanji_char} ({confidence:.1f}%)") 
        time.sleep(0.5)  # FPS optimization

thread = threading.Thread(target=inference_loop, daemon=True)
thread.start()

# =============================
# Main Video Loop
# =============================
cam_index = 0
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print("Cannot open camera")
    running = False
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        latest_frame = frame.copy()

    cv2.imshow('Kanji Live Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running = False
cap.release()
cv2.destroyAllWindows()
