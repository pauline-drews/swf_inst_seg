"""
03 Data Preparation

This is the 3rd step of the workflow for my Master thesis with the title: 

"Instance-Aware Semantic Segmentation of Small Woody Landscape Features 
 Using High-Resolution Aerial Images in Biodiversity Exploratories"

This script follows after script 02_dataset_generation.py and is followed 
by script 04_train_models.py.

In script 02_dataset_generation.py, a dataset of 5627 256x256 rgb-nir image
patches and the corresponding masks were created as .tif files from 300 
large aerial images and mask polygons of 3 classes of small woody 
features:
1: deciduous trees and tree groups,
2: coniferous trees and tree groups,
3: hedges <5 m height

In this script, the dataset is prepared for training in the following
steps:
1. Load dataset
2. Data sanity check
3. Image normalization and dimension adaptions
4. Create 6 mask variants and corresponding image_datasets:
**Theoretically, models could be run on all variants, but for this thesis, the 
temporal scope only allowed cases a) and f)**
   a) binary: all classes set to class 1
   b) class 1: only class 1 masks kept, class 2 and 3 set to background (0)
   c) class 2: equivalent to class 1
   d) class 3: equivalent to class 1 and class 2
   e) trees and hedges: classes 1 and 2 merged to "tree class" (1) and class 
      3 set to 2
   f) multi-class with all 3 classes
5. Visual check of random patches and masks for spatial alignment
6. Train-val-test split for all mask variants with 70% training data and each
   15 % validation and test data

In script 04_train_models, several segmentation models will be defined and 
trained with the data prepared here.
"""
###############################################################################
# 01 Load dataset
print("Loading dataset")

import os
import tifffile as tif
import numpy as np

data_path = "/data_hdd/pauline/swf/dataset/patches_256x256/"
image_path = os.path.join(data_path, "all_image_patches/")
mask_path = os.path.join(data_path, "all_mask_patches/")

image_dataset = []
mask_dataset = []
images = os.listdir(image_path)
masks = os.listdir(mask_path)

for img in range(1, len(images)):
    img_name = images[img]
    mask_name = masks[img]
    if img_name.endswith('.tif'):
        image = tif.imread(image_path + img_name)
        image_dataset.append(np.array(image))
    if mask_name.endswith('.tif'):
        mask = tif.imread(mask_path + mask_name)
        mask_dataset.append(np.array(mask))

image_dataset = np.array(image_dataset)
image_dataset = image_dataset.squeeze(1)
mask_dataset = np.array(mask_dataset)

print("Dataset loaded successfully with image dataset shape: ",
      image_dataset.shape, " and mask dataset shape: ",
      mask_dataset.shape)

###############################################################################
# 02 Data sanity check

print("Data sanity check:")

print("Image pixel value range: ", 
      np.min(image_dataset), " to ", 
      np.max(image_dataset))
      
# Check if there are any pixel values less than 0
if np.any(image_dataset < 0):
    # Replace pixel values less than 0 with 0.0
    image_dataset[image_dataset < 0] = 0.0
    print("Pixel values less than 0 found and updated to 0.0.")

# Print the updated pixel value range
print("Updated pixel value range: ", 
      np.min(image_dataset), " to ", 
      np.max(image_dataset))

nan_count = np.isnan(image).sum()
inf_count = np.isinf(image).sum()
print("Nan in images: ", nan_count, "; Inf in images: ", inf_count)

print("Mask pixel values: ", np.unique(mask_dataset))

###############################################################################
# 03 Image normalization and dimension adaptions

image_dataset = image_dataset / 255.0
print("Images normalized")

# add channel dimension to mask dataset
mask_dataset = np.expand_dims(np.array(mask_dataset), 3)

print("Shape of image dataset: ", image_dataset.shape)
print("Shape of mask dataset: ", mask_dataset.shape)

###############################################################################
# 04 Create 6 mask variants

print("Creating mask variants")

# a) binary: all classes set to class 1
masks_binary = np.where(mask_dataset != 0, 1, 0)
print("Created binary masks")

# b) class 1

def create_single_class_mask(mask_dataset, class_value):
    # Keep only the specified class value, setting all other values to 0
    single_class_mask = np.where(mask_dataset == class_value, 1, 0)
    return single_class_mask

masks_class1 = create_single_class_mask(mask_dataset, 1)
print("Created class 1 masks")

# c) class 2
masks_class2 = create_single_class_mask(mask_dataset, 2)
print("Created class 2 masks")

# d) class 3
masks_class3 = create_single_class_mask(mask_dataset, 3)
print("Created class 3 masks")

# e) trees vs. hedges
masks_trees_hedges = np.where(mask_dataset == 3, 2, 
                              np.where((mask_dataset == 1) | 
                                       (mask_dataset == 2), 1, 0))
print("Created trees vs. hedges masks")

# f) multi-class with all 3 classes
masks_multi_class = mask_dataset
print("Created multi-class masks")

print("All mask variants generated")

print("Creating image and mask datasets for all mask variants.")

# create the new dataset variants

# Function to calculate the non-zero pixel percentage
def calculate_non_zero_percentage(mask):
    return np.count_nonzero(mask) / mask.size * 100

# Function to filter image/mask pairs based on non-zero pixel percentage
def filter_dataset(image_dataset, mask_dataset, mask_variant, threshold=5):
    X_filtered = []
    y_filtered = []
    
    for img, mask in zip(image_dataset, mask_variant):
        if calculate_non_zero_percentage(mask) > threshold:
            X_filtered.append(img)
            y_filtered.append(mask)
    
    return np.array(X_filtered), np.array(y_filtered)

#  filter datasets
X_bin, y_bin = filter_dataset(image_dataset, mask_dataset, masks_binary)
X_class1, y_class1 = filter_dataset(image_dataset, mask_dataset, masks_class1)
X_class2, y_class2 = filter_dataset(image_dataset, mask_dataset, masks_class2)
X_class3, y_class3 = filter_dataset(image_dataset, mask_dataset, masks_class3)
X_trees_hedges, y_trees_hedges = filter_dataset(
    image_dataset, mask_dataset, masks_trees_hedges)
X_multiclass, y_multiclass = filter_dataset(
    image_dataset, mask_dataset, masks_multi_class)


print("Filtered datasets created:")
print(f"X_bin: {len(X_bin)}, y_bin: {len(y_bin)}")
print(f"X_class1: {len(X_class1)}, y_class1: {len(y_class1)}")
print(f"X_class2: {len(X_class2)}, y_class2: {len(y_class2)}")
print(f"X_class3: {len(X_class3)}, y_class3: {len(y_class3)}")
print(f"X_trees_hedges: {len(X_trees_hedges)}, 
      y_trees_hedges: {len(y_trees_hedges)}")
print(f"X_multiclass: {len(X_multiclass)}, 
      y_multiclass: {len(y_multiclass)}")

###############################################################################
# 05 Visual check of random patches and masks for spatial alignment
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# Define consistent colors for original classes
cmap_classes = ListedColormap([(0, 0, 0, 0),  # Class 0 (transparent)
                       (1, 0, 0, 0.5),  # Class 1 (red, semi-transparent)
                       (0, 1, 0, 0.5),  # Class 2 (green, semi-transparent)
                       (0, 0, 1, 0.5)]) # Class 3 (blue, semi-transparent)

cmap_binary = ListedColormap([(0, 0, 0, 0),  
                       (0, 0, 0, 0.9)]) 

cmap_trees_hedges = ListedColormap([(0, 0, 0, 0),  
                       (1, 0.5, 0, 0.5),
                       (0.5, 0, 0.5, 0.5)]) 


# visualize a random sample from a dataset
def get_random_image(X, y, mask_variant):
    idx = random.randint(0, len(X) - 1)
    image = X[idx]
    mask = y[idx]

    if mask_variant == "class2":
        mask =  np.where(mask == 1, 2, 0)
    if mask_variant == "class3":
        mask =  np.where(mask == 1, 3, 0)

    return image, mask


# Visualize random samples from all datasets
# a)
image, mask = get_random_image(X_bin, y_bin, "binary")
print(np.unique(mask))

plt.figure(figsize=(8, 8))
plt.imshow(image[..., :3]) 
plt.imshow(mask, cmap=cmap_binary, alpha=0.5)
plt.title(f"Random image and mask from binary masks")
plt.axis('off')
plt.show()

# b)
image, mask = get_random_image(X_class1, y_class1, "class1")
print(np.unique(mask))

plt.figure(figsize=(8, 8))
plt.imshow(image[..., :3]) 
plt.imshow(mask, cmap=cmap_classes, vmin=0, vmax=3)
plt.title(f"Random image and mask from class1 masks")
plt.axis('off')
plt.show()

# c)
image, mask = get_random_image(X_class2, y_class2, "class2")
print(np.unique(mask))

plt.figure(figsize=(8, 8))
plt.imshow(image[..., :3]) 
plt.imshow(mask, cmap=cmap_classes, vmin=0, vmax=3)
plt.title(f"Random image and mask from class2 masks")
plt.axis('off')
plt.show()

# d)
image, mask = get_random_image(X_class3, y_class3, "class3")
print(np.unique(mask))

plt.figure(figsize=(8, 8))
plt.imshow(image[..., :3]) 
plt.imshow(mask, cmap=cmap_classes, vmin=0, vmax=3)
plt.title(f"Random image and mask from class3 masks")
plt.axis('off')
plt.show()

# e)
image, mask = get_random_image(X_trees_hedges, y_trees_hedges, "trees_hedges")
print(np.unique(mask))

plt.figure(figsize=(8, 8))
plt.imshow(image[..., :3]) 
plt.imshow(mask, cmap=cmap_trees_hedges, vmin=0, vmax=3)
plt.title(f"Random image and mask from tree vs. hedges masks")
plt.axis('off')
plt.show()

# f)
image, mask = get_random_image(X_multiclass, y_multiclass, "multiclass")
print(np.unique(mask))

plt.figure(figsize=(8, 8))
plt.imshow(image[..., :3]) 
plt.imshow(mask, cmap=cmap_classes, vmin=0, vmax=3)
plt.title(f"Random image and mask from multiclass masks")
plt.axis('off')
plt.show()
###############################################################################

# 06 Train-val-test split for all mask variants
## improvement: keep names of original files (including plot name and number)
from sklearn.model_selection import train_test_split

# Function to split the datasets
def train_val_test_split(
        images, masks, val_size=0.15, test_size=0.15, random_state=42):
    # Split into train and remaining (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, masks, test_size=(val_size + test_size), 
        random_state=random_state
    )
    # Calculate test size proportion from val+test
    test_prop = test_size / (val_size + test_size)
    # Split remaining into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_prop, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Perform the split for each mask variant dataset
datasets = {
    "binary": (X_bin, y_bin),
    "class1": (X_class1, y_class1),
    "class2": (X_class2, y_class2),
    "class3": (X_class3, y_class3),
    "trees_hedges": (X_trees_hedges, y_trees_hedges),
    "multiclass": (X_multiclass, y_multiclass),
}

split_datasets = {}

for name, (X, y) in datasets.items():
    print(f"Splitting {name} dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    split_datasets[name] = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    print(f"{name} split: {len(X_train)} train, 
          {len(X_val)} val, {len(X_test)} test")

# Saving the datasets
output_dir = "/data_hdd/pauline/swf/dataset/patches_256x256/split_datasets/"
os.makedirs(output_dir, exist_ok=True)

for name, splits in split_datasets.items():
    for split_name, (X_split, y_split) in splits.items():
        split_dir = os.path.join(output_dir, f"{name}/{split_name}")
        os.makedirs(split_dir, exist_ok=True)
        # Save images
        image_dir = os.path.join(split_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        for idx, img in enumerate(X_split):
            tif.imwrite(os.path.join(
                image_dir, f"{name}_{split_name}_{idx}.tif"), img)
        # Save masks
        mask_dir = os.path.join(split_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        for idx, mask in enumerate(y_split):
            tif.imwrite(os.path.join(
                mask_dir, f"{name}_{split_name}_{idx}_mask.tif"), mask)

print("Train-val-test splits and saving completed.")