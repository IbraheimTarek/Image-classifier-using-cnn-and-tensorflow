import os
import random
from shutil import copyfile

# Define data directories
dataset_dir = 'dataset'
train_dir = 'train'
validation_dir = 'validation'
test_dir = 'test'

# Create train, validation, and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the split ratios
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Loop through each class folder
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        # Create corresponding class directories in train, validation, and test sets
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get list of images in the class folder
        images = os.listdir(class_dir)
        random.shuffle(images)
        
        # Split images into train, validation, and test sets
        num_train = int(train_ratio * len(images))
        num_validation = int(validation_ratio * len(images))
        
        train_images = images[:num_train]
        validation_images = images[num_train:num_train+num_validation]
        test_images = images[num_train+num_validation:]
        
        # Copy images to train directory
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_dir, class_name, img)
            copyfile(src, dst)
        
        # Copy images to validation directory
        for img in validation_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(validation_dir, class_name, img)
            copyfile(src, dst)
        
        # Copy images to test directory
        for img in test_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(test_dir, class_name, img)
            copyfile(src, dst)

# After splitting, proceed with the rest of the script (data preprocessing, model training, evaluation)
