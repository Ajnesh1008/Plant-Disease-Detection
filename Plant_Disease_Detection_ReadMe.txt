
🌿 Plant Disease Detection: ViT vs CNN
======================================

This project compares the performance of Vision Transformer (ViT) and Convolutional Neural Network (CNN) models for detecting plant diseases from images. It includes model training, evaluation, and a user-friendly desktop GUI to visualize predictions and performance.

📦 Project Structure
--------------------
.
├── compare_models.py        # Evaluates and compares both models with visual metrics
├── evaluate_models.py       # Saves performance metrics (accuracy, precision, recall, F1) to JSON
├── test.py                  # Tkinter-based GUI for real-time image classification
├── vit_plant_model.pth      # Trained ViT model weights
├── cnn_plant_model.pth      # Trained CNN model weights
├── class_names.json         # Class label mappings
├── model_metrics.json       # Output evaluation metrics for both models
└── dataset/
    └── test/                # Test images structured as: plant/disease/image.jpg

🚀 Features
-----------
- Model Comparison: Accuracy, precision, recall, F1-score, and inference time.
- Graphical User Interface (GUI): Upload an image and get instant predictions from both models.
- Visualization: Bar charts to compare accuracy and inference speed.
- Custom Dataset Compatible: Works with any image dataset following the folder structure.

🧠 Models Used
--------------
- Vision Transformer (ViT) from `timm`
- ResNet18 CNN from `torchvision`

Both models are fine-tuned on a custom dataset of plant diseases.

🖥️ Requirements
----------------
- Python 3.8+
- PyTorch
- torchvision
- timm
- scikit-learn
- matplotlib
- PIL (Pillow)
- Tkinter (pre-installed with most Python distributions)

Install dependencies:
pip install torch torchvision timm scikit-learn matplotlib Pillow

🔧 Usage
--------
1. Evaluate & Visualize
   python compare_models.py

2. Save Metrics to File
   python evaluate_models.py

   This will generate `model_metrics.json`.

3. Launch GUI
   python test.py

   This opens a Tkinter-based GUI where you can upload an image and compare the ViT and CNN predictions.

🗂️ Dataset Structure
---------------------
Ensure your dataset is structured like this:
dataset/
└── test/
    ├── Tomato/
    │   ├── healthy/
    │   └── leaf_mold/
    └── Potato/
        ├── healthy/
        └── early_blight/

📈 Output Metrics Example
--------------------------
From `model_metrics.json`:
{
  "vit": {
    "Accuracy": 0.97,
    "Precision": 0.96,
    "Recall": 0.95,
    "F1-Score": 0.95
  },
  "cnn": {
    "Accuracy": 0.94,
    "Precision": 0.91,
    "Recall": 0.90,
    "F1-Score": 0.90
  }
}

✨ Credits
----------
Developed as part of an AI project for plant disease detection using deep learning. Leveraging both classical CNN and modern Transformer architectures for robust classification.

📜 License
-----------
This project is for educational and research use. For commercial use, please contact the author.
