import os

# Dataset ka main path
DATASET_PATH = "dataset/raw"

# Teeno tumor categories
categories = ["Glioma", "Meningioma", "Pituitary tumor"]

for category in categories:

    folder_path = os.path.join(DATASET_PATH, category)

    # Total PNG files count karna
    files = os.listdir(folder_path)

    images = [f for f in files if f.endswith(".png") and not f.endswith("_mask.png")]

    masks = [f for f in files if f.endswith("_mask.png")]

    print(f"\nCategory: {category}")
    print(f"Total Images: {len(images)}")
    print(f"Total Masks: {len(masks)}")