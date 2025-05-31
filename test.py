import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models
from timm import create_model
import json
import os
import time

# ========================
# CONFIG
# ========================
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# TRANSFORMS
# ========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ========================
# LOAD CLASS NAMES
# ========================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ========================
# LOAD MODELS
# ========================
def load_models():
    # === Load ViT Model ===
    vit = create_model('vit_base_patch16_224', pretrained=False, num_classes=len(class_names))
    state_dict = torch.load("vit_plant_model.pth", map_location=DEVICE)

    # Remove classifier head to avoid mismatch
    for key in ["head.weight", "head.bias"]:
        if key in state_dict:
            del state_dict[key]

    vit.load_state_dict(state_dict, strict=False)
    vit.eval().to(DEVICE)

    # === Load CNN Model ===
    cnn = models.resnet18(weights=None)
    cnn.fc = torch.nn.Linear(cnn.fc.in_features, len(class_names))
    cnn.load_state_dict(torch.load("cnn_plant_model.pth", map_location=DEVICE))
    cnn.eval().to(DEVICE)

    return vit, cnn

vit_model, cnn_model = load_models()

# ========================
# PREDICT FUNCTION
# ========================
def predict(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = output.argmax(dim=1).item()
    end_time = time.time()

    label = class_names[pred_idx]
    plant, condition = label.split(" - ") if " - " in label else (label, "Unknown")
    status = "Healthy" if "healthy" in label.lower() else "Unhealthy"
    pred_time = end_time - start_time
    return plant, condition, status, pred_time

# ========================
# GUI LOGIC
# ========================
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    # Display image
    img = Image.open(file_path).resize((200, 200))
    tk_img = ImageTk.PhotoImage(img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Get predictions
    vit_plant, vit_condition, vit_status, vit_time = predict(file_path, vit_model)
    cnn_plant, cnn_condition, cnn_status, cnn_time = predict(file_path, cnn_model)

    # Update display
    vit_result.config(text=f"ViT: {vit_plant} - {vit_condition} ({vit_status})")
    cnn_result.config(text=f"CNN: {cnn_plant} - {cnn_condition} ({cnn_status})")
    vit_time_label.config(text=f"⏱️ ViT Prediction Time: {vit_time:.3f} sec")
    cnn_time_label.config(text=f"⏱️ CNN Prediction Time: {cnn_time:.3f} sec")

# ========================
# METRICS (STATIC PLACEHOLDER — replace with actual values)
# ========================
# vit_metrics = {"Accuracy": 0.97, "Precision": 0.96}
# cnn_metrics = {"Accuracy": 0.94, "Precision": 0.91}


try:
    with open("model_metrics.json", "r") as f:
        all_metrics = json.load(f)
        vit_metrics = all_metrics["vit"]
        cnn_metrics = all_metrics["cnn"]
except FileNotFoundError:
    vit_metrics = {"Accuracy": 0.0, "Precision": 0.0}
    cnn_metrics = {"Accuracy": 0.0, "Precision": 0.0}

# ========================
# GUI WINDOW
# ========================
root = tk.Tk()
root.title("🌿 Plant Disease Classifier (ViT vs CNN)")
root.geometry("600x600")
root.configure(bg="white")

tk.Label(root, text="Plant Disease Detection", font=("Arial", 16, "bold"), bg="white").pack(pady=10)
tk.Button(root, text="Select Image", command=browse_image, bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=5)

image_label = tk.Label(root, bg="white")
image_label.pack(pady=10)

vit_result = tk.Label(root, text="ViT: ", font=("Arial", 12), fg="green", bg="white")
vit_result.pack()

cnn_result = tk.Label(root, text="CNN: ", font=("Arial", 12), fg="blue", bg="white")
cnn_result.pack()

vit_time_label = tk.Label(root, text="", font=("Arial", 10), fg="gray", bg="white")
vit_time_label.pack()

cnn_time_label = tk.Label(root, text="", font=("Arial", 10), fg="gray", bg="white")
cnn_time_label.pack()

# Comparison summary
#tk.Label(root, text="Model Performance Summary", font=("Arial", 14, "underline"), bg="white").pack(pady=10)
#tk.Label(root, text=f"✅ ViT Accuracy: {vit_metrics['Accuracy']:.2f} | Precision: {vit_metrics['Precision']:.2f}", bg="white").pack()
#tk.Label(root, text=f"🔷 CNN Accuracy: {cnn_metrics['Accuracy']:.2f} | Precision: {cnn_metrics['Precision']:.2f}", bg="white").pack()
tk.Label(root, text=f"✅ ViT Accuracy: {vit_metrics['Accuracy']:.2f} | Precision: {vit_metrics['Precision']:.2f}", bg="white").pack()
tk.Label(root, text=f"🔷 CNN Accuracy: {cnn_metrics['Accuracy']:.2f} | Precision: {cnn_metrics['Precision']:.2f}", bg="white").pack()



root.mainloop()
