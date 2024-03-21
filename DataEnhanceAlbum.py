import os
import cv2
import numpy as np
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, GaussNoise, Normalize
)
from albumentations.pytorch import ToTensorV2

# Define augmentation sequence
augmentations = Compose([
    HorizontalFlip(p=0.5),  # 50% probability of horizontal flip
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Random brightness and contrast
    GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Add Gaussian noise
    # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize
])

# Input and output folder paths
input_folder_path = 'data/Processed224x224data/1'
output_folder_path = 'data/Augmented224x224data_albumentations'

# Ensure output folder exists
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Iterate through images in the input folder, augment them, and save them
for filename in os.listdir(input_folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check the file extension
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(output_folder_path, f"aug_{filename}")
        
        # Read the image
        image = cv2.imread(input_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
        
        # Apply augmentation sequence
        augmented = augmentations(image=image)
        image_aug = augmented['image']
        
        # Convert image from RGB to BGR format for saving
        image_aug_bgr = cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR)
        
        # Save the augmented image
        cv2.imwrite(output_file_path, image_aug_bgr)
        
        print(f"Processed and saved: {output_file_path}")
        
print("All images have been processed and saved.")