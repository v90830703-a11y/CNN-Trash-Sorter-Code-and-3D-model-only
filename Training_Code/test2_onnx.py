import onnxruntime as ort
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "Training_Code/Tmodel.onnx"
TEST_DIR = r"Training_Code/Data_Splitted/test"
IMG_SIZE = 96
BATCH_SIZE = 32  # faster, still correct

# ----------------------------
# LOAD ONNX MODEL
# ----------------------------
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# ----------------------------
# IMPORTANT: MATCH TRAINING EXACTLY
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),

    # 🔥 ADD THIS ONLY IF USED IN TRAINING
    # Example (VERY COMMON):
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# ----------------------------
# DATA
# ----------------------------
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", test_dataset.classes)
print("Total test images:", len(test_dataset))

# ----------------------------
# EVAL
# ----------------------------
correct = 0
total = 0

all_preds = []
all_labels = []

for images, labels in test_loader:

    # PyTorch -> NumPy
    images = images.numpy().astype(np.float32)

    # ONNX inference
    outputs = session.run(None, {input_name: images})[0]

    # logits -> prediction
    preds = np.argmax(outputs, axis=1)

    correct += (preds == labels.numpy()).sum()
    total += labels.size(0)

    all_preds.extend(preds)
    all_labels.extend(labels.numpy())

# ----------------------------
# RESULTS
# ----------------------------
acc = correct / total

print("\n===== RESULTS =====")
print("Accuracy:", acc)
print("Correct:", correct)
print("Total:", total)

# Confusion matrix
from sklearn.metrics import confusion_matrix
import collections

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)

print("\nPrediction distribution:", collections.Counter(all_preds))