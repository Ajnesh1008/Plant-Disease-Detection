import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from torch.optim import Adam
from tqdm import tqdm
from PIL import Image

# ==============================
# CONFIG
# ==============================
DATA_DIR = "dataset"
BATCH_SIZE = 16
EPOCHS = 10
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = len(os.listdir(os.path.join(DATA_DIR, "train")))

# ==============================
# TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ==============================
# DATA LOADERS
# ==============================
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==============================
# MODEL
# ==============================
model = create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# ==============================
# LOSS & OPTIMIZER
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()

    acc = train_correct / len(train_dataset)
    print(f"✅ Epoch {epoch+1} | Loss: {train_loss:.4f} | Accuracy: {acc:.4f}")

# ==============================
# SAVE MODEL & CLASS NAMES
# ==============================
torch.save(model.state_dict(), "vit_plant_model.pth")
with open("class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)
print("✅ Model and class names saved.")


# ==============================
# PREDICTION BLOCK (Example)
# ==============================

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load model
model.load_state_dict(torch.load("vit_plant_model.pth", map_location=DEVICE))
model.eval()

# Predict sample image
sample_path = "sample_leaf.jpeg"  # Replace with your test image path
if os.path.exists(sample_path):
    img = Image.open(sample_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    output = model(img_tensor)
    predicted_class_idx = output.argmax(dim=1).item()
    predicted_label = class_names[predicted_class_idx]

    # Parse label
    if " - " in predicted_label:
        plant, condition = predicted_label.split(" - ", 1)
    else:
        plant, condition = predicted_label, "Unknown"

    # Determine health
    health_status = "Healthy" if "healthy" in predicted_label.lower() else "Unhealthy"

    # Print results
    print("\n📷 Prediction Results")
    print(f"🌿 Plant: {plant}")
    print(f"🩺 Condition: {condition}")
    print(f"✅ Health Status: {health_status}")
else:
    print("\n⚠️ Sample image not found. Place a test image at: sample_leaf.jpg")