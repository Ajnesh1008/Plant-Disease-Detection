# evaluate_models.py
import torch
from torchvision import datasets, transforms, models
from timm import create_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Data loader
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_dataset = datasets.ImageFolder("dataset/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load models
def load_model(model_type, path):
    if model_type == "vit":
        model = create_model("vit_base_patch16_224", pretrained=False, num_classes=len(class_names))
        state_dict = torch.load(path, map_location=DEVICE)
        for key in ["head.weight", "head.bias"]:
            if key in state_dict:
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    else:  # cnn
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

vit_model = load_model("vit", "vit_plant_model.pth")
cnn_model = load_model("cnn", "cnn_plant_model.pth")

# Evaluation
def evaluate_model(model):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

vit_metrics = evaluate_model(vit_model)
cnn_metrics = evaluate_model(cnn_model)

# Save to file
with open("model_metrics.json", "w") as f:
    json.dump({"vit": vit_metrics, "cnn": cnn_metrics}, f, indent=4)

print("✅ Evaluation done. Metrics saved to model_metrics.json.")
