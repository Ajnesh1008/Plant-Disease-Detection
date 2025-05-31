import os
import time
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from timm import create_model
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import json

# ==============================
# CONFIG
# ==============================
DATA_DIR = "dataset"
IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# LOAD CLASS NAMES
# ==============================
with open("class_names.json", "r") as f:
    class_names = json.load(f)
NUM_CLASSES = len(class_names)

# ==============================
# TRANSFORM & TEST DATA
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.2], [0.2])
])
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==============================
# EVALUATION FUNCTION
# ==============================
def evaluate_model(model, model_name):
    model.eval()
    y_true, y_pred = [], []
    total_time = 0
    total_images = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            preds = torch.argmax(outputs, dim=1).cpu()
            y_pred.extend(preds.numpy())
            y_true.extend(labels.numpy())

            total_time += (end_time - start_time)
            total_images += inputs.size(0)

    avg_image_time = total_time / total_images
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    print(f"\n📊 Results for {model_name}:")
    print(f"✅ Accuracy           : {acc:.4f}")
    print(f"🎯 Precision          : {precision:.4f}")
    print(f"🔁 Recall             : {recall:.4f}")
    print(f"🏅 F1 Score           : {f1:.4f}")
    print(f"⚡ Avg Time per Image : {avg_image_time:.6f} sec")

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_time_image": avg_image_time
    }

# ==============================
# LOAD ViT MODEL
# ==============================
vit_model = create_model('vit_base_patch16_224', pretrained=False, num_classes=17)
vit_model.load_state_dict(torch.load("vit_plant_model.pth", map_location=DEVICE))
vit_model.to(DEVICE)

# ==============================
# LOAD CNN MODEL
# ==============================
cnn_model = models.resnet18(weights=None)
cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, NUM_CLASSES)
cnn_model.load_state_dict(torch.load("cnn_plant_model.pth", map_location=DEVICE))
cnn_model.to(DEVICE)

# ==============================
# EVALUATE BOTH MODELS
# ==============================
vit_metrics = evaluate_model(vit_model, "ViT")
cnn_metrics = evaluate_model(cnn_model, "CNN")

# ==============================
# ACCURACY BAR CHART
# ==============================
models_list = [vit_metrics["model"], cnn_metrics["model"]]
accuracies = [vit_metrics["accuracy"], cnn_metrics["accuracy"]]

plt.figure(figsize=(6, 4))
bars = plt.bar(models_list, accuracies, color=['skyblue', 'salmon'])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison (ViT vs CNN)")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02, f'{height:.2f}', ha='center')

plt.tight_layout()
plt.show()

# ==============================
# PREDICTION EFFICIENCY BAR CHART
# ==============================
prediction_times = [vit_metrics["inference_time_image"], cnn_metrics["inference_time_image"]]

plt.figure(figsize=(6, 4))
bars = plt.bar(models_list, prediction_times, color=['purple', 'darkcyan'])
plt.ylabel("Avg Inference Time per Image (s)")
plt.title("Prediction Efficiency (Lower is Better)")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.0001, f'{height:.6f}s', ha='center')

plt.tight_layout()
plt.show()
