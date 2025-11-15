import cv2
import torch
import torch.nn as nn
import threading
import time
import numpy as np
from PIL import Image
from torchvision import transforms

# =============================
# Labels for Kuzushiji-46
# =============================
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
    "わ", "を",
    "ん"
]

torch.set_num_threads(1)

# =============================
# Device Detection
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device}")

# =============================
# CNN Model
# =============================
class KanjiCNN(nn.Module):
    def __init__(self, num_classes=46):
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
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = KanjiCNN().to(device)
model.load_state_dict(torch.load("best_kanji_cnn_model.pth", map_location=device))
model.eval()

# =============================
# Transform (Same as Training)
# =============================
# Load your training images
imgs_npz = np.load("./data/hiragana_final/hiragana-train-imgs.npz")
images = imgs_npz[imgs_npz.files[0]]  # Assuming first key contains image data

# Compute mean and std across all pixels
mean = images.mean() / 255.0  # Normalize to [0,1] range if original is 0-255
std = images.std() / 255.0

inference_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# =============================
latest_frame = None
prediction_text = "Loading..."
lock = threading.Lock()
running = True

# =============================
# Inference Thread (Corrected)
# =============================

from deep_translator import GoogleTranslator

# =============================
# Enhanced Multi-Character Parsing
# =============================
def parse_multiple_characters(frame, target_size=28):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return ""

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    assembled_text = []
    count = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue

        crop = thresh[y:y+h, x:x+w]
        h_c, w_c = crop.shape
        side = max(w_c, h_c)
        square = np.full((side, side), 0, dtype=np.uint8)
        x_offset = (side - w_c) // 2
        y_offset = (side - h_c) // 2
        square[y_offset:y_offset+h_c, x_offset:x_offset+w_c] = crop

        white_ratio = np.sum(square == 255) / (square.shape[0] * square.shape[1])
        kernel = np.ones((2, 2), np.uint8)
        if white_ratio > 0.25:
            square = cv2.erode(square, kernel, iterations=1)

        resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
        resized = cv2.GaussianBlur(resized, (3, 3), 0)
        resized = cv2.dilate(resized, kernel, iterations=1)
        noise = np.random.normal(0, 2, resized.shape).astype(np.float32)
        resized = resized.astype(np.float32) + noise
        resized = np.clip(resized, 0, 255).astype(np.uint8)
        cv2.imwrite(f"test_images/test_image{count}.jpg", resized)
        count += 1

        img_pil = Image.fromarray(resized)
        tensor = inference_transform(img_pil).unsqueeze(0).to(device)
        
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item() * 100
            pred_label = labels[pred_idx]

        # Only add if confidence is above threshold
        if confidence >= 40:  # You can adjust this threshold
            assembled_text.append(pred_label)

    return "".join(assembled_text)

# =============================
# New Inference Loop
# =============================
def inference_loop():
    global latest_frame, running
    while running:
        if latest_frame is not None:
            with lock:
                frame_copy = latest_frame.copy()

            hiragana_string = parse_multiple_characters(frame_copy)
            if hiragana_string:
                print(f"Detected Hiragana: {hiragana_string}")
                try:
                    translation = GoogleTranslator(source='ja',target='en').translate(hiragana_string)
                    print(f"Translation: {translation}")
                except Exception as e:
                    print(f"Translation error: {e}")

        time.sleep(1)

thread = threading.Thread(target=inference_loop, daemon=True)
thread.start()

# =============================
# Main Video Loop
# =============================
cap = cv2.VideoCapture(0)
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
