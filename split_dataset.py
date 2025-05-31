import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
base_dir = "dataset"
test_dir = os.path.join(base_dir, "test")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Clean or create train and val dirs
for d in [train_dir, val_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# Process each plant type and class
for plant in os.listdir(test_dir):
    plant_path = os.path.join(test_dir, plant)
    if not os.path.isdir(plant_path):
        continue

    for condition in os.listdir(plant_path):
        class_path = os.path.join(plant_path, condition)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Step 1: Keep 20% in test, 80% for further split
        test_images, rest_images = train_test_split(images, test_size=0.8, random_state=42)

        # Step 2: From remaining 80%, split into 80% train and 20% val
        train_images, val_images = train_test_split(rest_images, test_size=0.2, random_state=42)

        # Create destination folders
        def make_dirs(subdir):
            dir_path = os.path.join(subdir, plant, condition)
            os.makedirs(dir_path, exist_ok=True)
            return dir_path

        train_dest = make_dirs(train_dir)
        val_dest = make_dirs(val_dir)

        # Move files
        for img in train_images:
            shutil.move(os.path.join(class_path, img), os.path.join(train_dest, img))
        for img in val_images:
            shutil.move(os.path.join(class_path, img), os.path.join(val_dest, img))
        # Test images remain in place

print("✅ Split complete: 20% test | 64% train | 16% val")
