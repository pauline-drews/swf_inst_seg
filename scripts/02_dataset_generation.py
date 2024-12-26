""" 
02 Dataset Generation

This is the 2nd step of the workflow for my Master thesis with the title:

"Instance-Aware Semantic Segmentation of Small Woody Landscape Features Using 
 High-Resolution Aerial Images in Biodiversity Exploratories"

This script follows after script 01_extract_images.R and is followed by 
script 03_data_preparation.py.

In script 01_extract_images.R, aerial images were extracted from the RSDB 
database. For the training dataset, 300 images were extracted, from 50 grass-
land and 50 forest plots each in 3 exploratories. The images have a size of 
3535x3535 pixels, or 706.8x706.8 meters, which corresponds to the size of the 
largest square fitting into the circle plots of 100 m diameter. The resolution 
of the images is 0.02 m.

In this script, a dataset of 256x256 image patches and corresponding masks is 
generated from the large aerial images and from mask polygons, which have been 
created from the ATKIS dataset 
(https://pad.gwdg.de/lsm_classification_key#Updating-the-ATKIS-layers) and an 
updated ATKIS dataset 
(https://pad.gwdg.de/lsm_classification_key#3-Agricultural-land). 
The polygons represent small woody landscape features (SWFs) of three classes: 
1: broadleaf trees and tree groups; 2: coniferous trees and tree groups; 3: 
hedges < 5 m. 
The following steps are performed in this script:

1. Input Preparation:
   - Large aerial images (4 channels: RGB + NIR, 32 bit) are loaded
   - Mask polygons (stored in a shapefile) are loaded

2. Polygon Rasterization:
   - Polygons are rasterized into masks with priority handling for overlaps 
     (coniferous trees > deciduous trees > hedges).

3. Large Mask Creation:
   - Large masks corresponding to the aerial images are created and saved as 
     .tif files.

4. Patch Extraction:
   - Images and masks are divided into 256x256 patches with overlap to ensure 
     coverage of all areas of the large images
   - Only patches with more than 5% non-zero pixels in their masks are saved to 
     avoid storing patches with low information value.

    Sanity Checks:
        Random samples of images and their corresponding masks are visualized 
        to verify spatial alignment and proper overlay of classes.

In script 03_data_preparation.py, this dataset will be further processed, where 
mask variants will be created, and the dataset will be prepared for training 
and split to training, validation and test subsets. 
"""

# Set the random seed for reproducibility
import random
import numpy as np

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# Set new patch dimensions
patch_size = 256
patch_overlap = 25
effective_patch_size = patch_size - patch_overlap

# Define input paths
image_dir = '/data_hdd/pauline/swf/dataset/largest_square_plot_images/'
shapefile_dir = '/data_hdd/pauline/swf/dataset/filtered_polygons/'
# Define output directories for saving patches and masks
import os

image_patch_output_dir = '/data_hdd/pauline/swf/dataset/patches_256x256/all_image_patches/'
large_mask_output_dir = '/data_hdd/pauline/swf/dataset/largest_square_plot_masks/'
mask_patch_output_dir = '/data_hdd/pauline/swf/dataset/patches_256x256/all_image_masks/'
os.makedirs(image_patch_output_dir, exist_ok=True)
os.makedirs(large_mask_output_dir, exist_ok=True)
os.makedirs(mask_patch_output_dir, exist_ok=True)

# Map class_1 values to mask values
class_mapping = {51: 1, 52: 2, 53: 3}
priority_classes = [52, 51, 53]  # Priority order for overlap handling

###############################################################################
# 01 Input Preparation

# Get all image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

# Read the polygons from the shapefile
import fiona
from shapely.geometry import shape, box

# Read the polygons from the shapefile
polygons = []
with fiona.open(shapefile_dir, 'r') as shp:
    for feature in shp:
        geom = shape(feature['geometry'])
        class_1 = feature['properties']['class_1']
        if class_1 in class_mapping:
            polygons.append((geom, class_1))

###############################################################################
# 02 Polygon Rasterization
# 03 Large Mask Creation

# Function to rasterize polygons with priority handling
def rasterize_with_priority(polygons, image_bounds, image_shape, transform):
    # Initialize mask with zeros
    mask = np.zeros(image_shape, dtype=np.uint8)
    for priority_class in priority_classes:
        # Filter polygons for the current priority class
        priority_polygons = [
            (geom, class_mapping[class_1]) for geom, 
            class_1 in polygons if class_1 == priority_class]
        if priority_polygons:
            # Rasterize current priority class polygons
            rasterized = rasterize(
                [(geom, value) for geom, value in priority_polygons],
                out_shape=image_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            # Update mask, but only overwrite where it is currently zero
            mask = np.where((mask == 0) & (rasterized > 0), rasterized, mask)
    return mask

# Process each image
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    with rasterio.open(image_path) as src:
        # Get image properties
        image_bounds = src.bounds
        image_shape = (src.height, src.width)
        transform = from_bounds(*image_bounds, src.width, src.height)

        # Filter polygons that intersect the image bounds
        intersecting_polygons = [
            (geom, class_1) for geom, 
            class_1 in polygons if geom.intersects(box(*image_bounds))]

        # Rasterize polygons to create the mask
        mask = rasterize_with_priority(
            intersecting_polygons, image_bounds, image_shape, transform)

        # Save the mask as a TIFF file
        mask_path = os.path.join(
            large_mask_output_dir, 
            f"{os.path.splitext(image_file)[0]}_mask.tif")
        with rasterio.open(
            mask_path,
            'w',
            driver='GTiff',
            height=image_shape[0],
            width=image_shape[1],
            count=1,
            dtype=np.uint8,
            crs=src.crs,
            transform=transform,
        ) as dst:
            dst.write(mask, 1)

print("Large masks have been created and saved.")

# Sanity check
mask_files = sorted(
    [f for f in os.listdir(large_mask_output_dir) if f.endswith('_mask.tif')])
image_files = sorted(
    [f for f in os.listdir(image_dir) if f.endswith('.tif')])

image_basenames = [os.path.splitext(f)[0] for f in image_files]
mask_basenames = [
    os.path.splitext(f)[0].replace('_mask', '') for f in mask_files]
common_basenames = list(set(image_basenames) & set(mask_basenames))
selected_basename = random.choice(common_basenames)
image_path = os.path.join(image_dir, f"{selected_basename}.tif")
mask_path = os.path.join(
    large_mask_output_dir, f"{selected_basename}_mask.tif")


# Load image and mask
with rasterio.open(image_path) as img_src:
    image = img_src.read([1, 2, 3]).transpose(1, 2, 0)  # Read RGB channels
    image = np.clip(image / image.max(), 0, 1)  # Normalize for plotting

with rasterio.open(mask_path) as mask_src:
    mask = mask_src.read(1)

# Define colors and transparency for the mask
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap([(0, 0, 0, 0),  # Class 0 (transparent)
                       (1, 0, 0, 0.5),  # Class 1 (red, semi-transparent)
                       (0, 1, 0, 0.5),  # Class 2 (green, semi-transparent)
                       (0, 0, 1, 0.5)])  # Class 3 (blue, semi-transparent)

# Plot the image and mask
plt.figure(figsize=(10, 10))
plt.imshow(image, interpolation='nearest')
plt.imshow(mask, cmap=cmap, interpolation='nearest', alpha=0.7)  # Overlay mask
plt.title(f"Image and Mask Overlay: {selected_basename}")
plt.axis('off')
plt.show()

###############################################################################
# 04 Patch Extraction

# Generate 256x256 patches for each image and corresponding mask
from patchify import patchify
import tifffile as tiff

def get_nonzero_percentage(mask_patch):
    """Calculate the percentage of non-zero pixels in a mask patch."""
    total_pixels = mask_patch.size
    non_zero_pixels = np.count_nonzero(mask_patch)
    perc_non_zero = (non_zero_pixels / total_pixels) * 100
    return perc_non_zero

def process_images_and_masks(image_dir, 
                             mask_dir, 
                             image_patch_output, 
                             mask_patch_output, 
                             patch_size):
    """Cut images & masks into patches & save them based on specified criteria."""
    os.makedirs(image_patch_output, exist_ok=True)
    os.makedirs(mask_patch_output, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    
    for image_file in image_files:
        # Construct the corresponding mask filename
        base_name = image_file.replace("squ_", "").replace(".tif", "")  # Remove "squ_" and get base name
        mask_file = f"squ_{base_name}_mask.tif"  # Construct the correct mask filename
        mask_path = os.path.join(mask_dir, mask_file)

        # Check if the mask file exists
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            continue

        # Read image and mask
        image = tiff.imread(os.path.join(image_dir, image_file))
        mask = tiff.imread(mask_path)

        # # Add padding
        # # Padding values
        # top_left_pad = 24
        # bottom_right_pad = 25

        # # Pad the image (RGB + NIR)
        # padded_image = np.pad(
        #     image,
        #     pad_width=(
        #         (top_left_pad, bottom_right_pad), 
        #         (top_left_pad, bottom_right_pad), (0, 0)),
        #     mode='constant',
        #     constant_values=0.0
        #     )

        # # Pad the mask (multiclass)
        # padded_mask = np.pad(
        #     mask,
        #     pad_width=(
        #         (top_left_pad, bottom_right_pad), 
        #         (top_left_pad, bottom_right_pad)),
        #     mode='constant',
        #     constant_values=0
        #     )
        
        # print("padded image and mask shapes: ", 
        #       padded_image.shape, padded_mask.shape)

        # Patchify the image and mask
        image_patches = patchify(
            image, (patch_size, patch_size, 4), step=patch_size)
        mask_patches = patchify(
            mask, (patch_size, patch_size), step=patch_size)


        patch_count = 0  # Reset patch count for each new image
        
        # Iterate through patches
        for i in range(image_patches.shape[0]):
            for j in range(image_patches.shape[1]):
                image_patch = image_patches[i, j, :, :, :]
                mask_patch = mask_patches[i, j, :, :]

                # Check for non-zero pixels in the mask patch
                if get_nonzero_percentage(mask_patch) > 5:  # More than 5% non-zero pixels
                    # Create output filenames
                    patch_number = f"{patch_count:03d}"  # Format patch number with leading zeros
                    image_patch_name = f"image_patch_{base_name}_{patch_number}.tif"
                    mask_patch_name = f"mask_patch_{base_name}_{patch_number}.tif"

                    # Save the patches
                    tiff.imwrite(
                        os.path.join(image_patch_output, 
                                     image_patch_name), 
                                     image_patch)
                    tiff.imwrite(os.path.join(mask_patch_output, 
                                              mask_patch_name), 
                                              mask_patch)

                    # Print saved patches information
                    print(f"Saved: {image_patch_name} and {mask_patch_name}")

                    patch_count += 1

process_images_and_masks(image_dir, 
                         large_mask_output_dir, 
                         image_patch_output_dir, 
                         mask_patch_output_dir, 
                         patch_size)


###############################################################################
# Sanity check
mask_files = sorted(
    [f for f in os.listdir(mask_patch_output_dir) if f.endswith('.tif')])
image_files = sorted(
    [f for f in os.listdir(image_patch_output_dir) if f.endswith('.tif')])

image_basenames = [
    os.path.splitext(f)[0].replace('image_', '') for f in image_files]
mask_basenames = [
    os.path.splitext(f)[0].replace('mask_', '') for f in mask_files]
common_basenames = list(set(image_basenames) & set(mask_basenames))
selected_basename = random.choice(common_basenames)
# Construct the paths to the selected image and mask
image_path = os.path.join(
    image_patch_output_dir, ("image_" + f"{selected_basename}.tif"))
mask_path = os.path.join(
    mask_patch_output_dir, ("mask_" + f"{selected_basename}.tif"))

# Load image and mask
import rasterio
with rasterio.open(image_path) as img_src:
    image = img_src.read([1, 2, 3]).transpose(1, 2, 0)  # Read RGB channels
    image = np.clip(image / image.max(), 0, 1)  # Normalize for plotting

with rasterio.open(mask_path) as mask_src:
    mask = mask_src.read(1)
    unique_values = np.unique(mask)
    print(unique_values)

# If the mask has only one unique value, add a random pixel with value 0
if len(unique_values) == 1:
    height, width = mask.shape
    random_y = np.random.randint(0, height)
    random_x = np.random.randint(0, width)
    mask[random_y, random_x] = 0  # Add a random pixel with value 0

# Define colors and transparency for the mask
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap([(0, 0, 0, 0),  # Class 0 (transparent)
                       (1, 0, 0, 0.5),  # Class 1 (red, semi-transparent)
                       (0, 1, 0, 0.5),  # Class 2 (green, semi-transparent)
                       (0, 0, 1, 0.5)])  # Class 3 (blue, semi-transparent)

# Plot the image and mask
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.imshow(mask, cmap=cmap)  # Overlay mask
plt.title(f"Image and Mask Overlay: {selected_basename}")
plt.axis('off')
plt.show()