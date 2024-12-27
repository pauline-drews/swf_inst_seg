'''
06 Model Prediction and Watershed

This is the 6th step of the workflow for my Master’s thesis with the title:

"Instance-Aware Semantic Segmentation of Small Woody Landscape Features
Using High-Resolution Aerial Images in Biodiversity Exploratories"

Parts of this script were taken and modified from the tutorial script of
Dr. Sreenivas Bhattiprolu:
https://github.com/bnsreenu/python_for_microscopists/blob/master/205_predict_unet_with_watershed_single_image.py

This script follows after 05_model_evaluation.py.

In script 05_model_evaluation.py, the performance of the trained models was 
quantified on testing data using IoU for the binary case and mean IoU &
per-class IoU for the multi-class case.

In this script, the trained models are used to predict segmentation masks for 
large aerial images. Predictions are performed using both a simple U-Net and a 
residual U-Net with attention for binary and multi-class segmentation cases. 
These predictions are enhanced to achieve instance awareness by applying the 
Watershed algorithm. The process involves:

1. Predicting masks in patches, reconstructing them for the original image 
dimensions, and saving the results.
2. Visualizing predictions overlaid on the original aerial images.
3. Applying the Watershed algorithm to segment small woody landscape features 
into individual instances.
4. Saving watershed results as labeled images and extracting statistics for 
each instance (e.g., area, solidity, equivalent diameter).
5. Generating plots of predictions, watershed results, and overlays with the 
original images and masks.
6. Statistical analysis of watershed statistics
'''

from simple_unet_model import simple_unet_model
from simple_multi_unet_model import multi_unet_model
from res_unet_attention import Attention_ResUNet, jacard_coef
from focal_loss import BinaryFocalLoss
from tensorflow.keras.optimizers import Adam
import keras
import tensorflow as tf
import os
import tifffile as tiff
import numpy as np
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as ptchs
import cv2
from skimage import measure, color
import pandas as pd
import seaborn as sns

# General parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 4
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Paths
model_path = "/data_hdd/pauline/swf/results/model_weights/b_final_models/"
large_img_dir = "/data_hdd/pauline/swf/dataset/large_prediction_images/"
large_masks_dir = '/data_hdd/pauline/swf/dataset/largest_square_plot_masks/'
pred_mask_output_dir = '/data_hdd/pauline/swf/results/segmentation_prediction_masks/'
watershed_plot_output_dir = 'data_hdd/pauline/swf/results/watershed/instance_segmentation_plots'
watershed_stats_output_dir = 'data_hdd/pauline/swf/results/watershed/stats/'
pred_plots_dir = "/data_hdd/pauline/swf/results/segmentation_plots/"

# color maps for plots
bin_cmap = ListedColormap(
    [(1, 1, 1, 0), # Transparent for 0
     (1, 0.5, 0, 0.35)]) # semi-transparent orange for 1

multi_cmap = ListedColormap(
    [(0, 0, 0, 0),  # Class 0 (transparent)
    (1, 0, 0, 0.35),  # Class 1 (red, semi-transparent)
    (0, 1, 0, 0.35),  # Class 2 (green, semi-transparent)
    (0, 0, 1, 0.35)])  # Class 3 (blue, semi-transparent)

mc_watershed_colors = {
    1: (1, 0, 0, 0.35),  # Red with alpha 0.35
    2: (0, 1, 0, 0.35),  # Green with alpha 0.35
    3: (0, 0, 1, 0.35),  # Blue with alpha 0.35
    -1: (255, 255, 0, 1),# Yellow with alpha 1 (no transparency)
}

###############################################################################
'''01 Predicting masks in patches, reconstructing them for the original image 
      dimensions, and saving the results
   02 Visualizing predictions overlaid on the original aerial images
   03 Applying the Watershed algorithm to segment small woody landscape 
      features into individual instances
   04 Saving watershed results as labeled images and extracting statistics for 
      each instance (e.g., area, solidity, equivalent diameter)
   05 Generating plots of predictions, watershed results, and overlays with the 
      original images and masks.'''
###############################################################################

# List all image files in the directory
image_files = [f for f in os.listdir(large_img_dir) if f.endswith('.tif')]

# Loop through all images for predictions of all models
for img_name in image_files:
    # load image and mask
    img_n = img_name.replace("squ_", "").replace(".tif", "")
    print(f"Processing {img_n}...")
    mask_name = img_name.replace(".tif", "_mask.tif")
    large_image = tiff.imread(os.path.join(large_img_dir, img_name))
    mask = tiff.imread(os.path.join(large_masks_dir, mask_name))
    mask_bin = np.where(mask != 0, 1, 0)

    # Pad the image such that patches can be drawn without rest pixels
    step_size = 250
    padded_image = np.pad(
        large_image,
        pad_width=((3, 2), (3, 2), (0, 0)),  # Specify padding for dimensions
        mode='constant',  # Use constant padding with default value 0
    )
    patches = patchify(padded_image, (256, 256, 4), step=step_size)

    # Normalize image
    large_image_norm = large_image / 255.0

    # create padding for original mask to fit image dimensions and for plot box
    large_height, large_width, _ = large_image.shape
    mask_height, mask_width = mask.shape
    pad_top = (large_height - mask_height) // 2
    pad_bottom = large_height - mask_height - pad_top
    pad_left = (large_width - mask_width) // 2
    pad_right = large_width - mask_width - pad_left
    if (large_height - mask_height) % 2 != 0:
        pad_top += 1
    if (large_width - mask_width) % 2 != 0:
        pad_right += 1

    # Save the image alone with area where masks exist
    plt.figure(figsize=(8, 8))
    plt.title(f'{img_n}')
    plt.imshow(large_image_norm[:, :, :3]) 
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none', zorder=3
    )
    ax.add_patch(rect)
    plt.axis('off')  # Remove axis for clean saving
    plt.savefig(os.path.join(
            pred_plots_dir,
            f'large_pred_image_{img_n}.png'), 
        bbox_inches='tight', dpi=300)
    plt.show()

    # pad mask to fit large image dimensions
    padded_mask = np.pad(
        mask_bin,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    # plot original image with existing mask part
    plt.figure(figsize=(8, 8))
    plt.title(f'Binary mask on {img_n}')
    plt.imshow(large_image_norm[:, :, :3]) 
    plt.imshow(padded_mask, cmap=bin_cmap)  # Overlay the padded mask
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(os.path.join(
            pred_plots_dir,
            f'mask_of_{img_n}_bin.png'), 
        bbox_inches='tight', dpi=300)
    plt.show()

    # Prediction
    n_classes = 2

###############################################################################
    '''
    Simple U-Net
    '''

    def get_simple_unet_model():
        return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model_name = "simp_unet_bin_50ep_B16_binCEloss.hdf5"
    model_n = model_name.replace(".hdf5", "")
    simple_unet_bin = get_simple_unet_model()
    simple_unet_bin.load_weights(os.path.join(model_path, model_name))

    # predict in patches of size that model was trained with
    predicted_patches = []
    count = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :, :, :]
            single_patch_norm = single_patch / 255.0
            single_patch_prediction = (
                simple_unet_bin.predict(
                    single_patch_norm)[0, :, :, 0] > 0.5).astype(np.uint8)
            predicted_patches.append(single_patch_prediction)

    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(
        predicted_patches,
        (patches.shape[0], patches.shape[1], IMG_HEIGHT, IMG_WIDTH)
    )
    reconstructed_prediction = unpatchify(
        predicted_patches_reshaped,
        padded_image.shape[:2]
    )
    unpadded_prediction = reconstructed_prediction[3:-2, 3:-2]

    save_path = os.path.join(
        pred_mask_output_dir, 
        f"{img_n}_pred_mask_{model_n}.tif")
    tiff.imwrite(
        save_path, 
        unpadded_prediction.astype(np.uint8))

    # Save the image with prediction overlay
    plt.figure(figsize=(8, 8))
    plt.title(f'Prediction {img_n} & {model_n}')
    plt.imshow(large_image_norm[:, :, :3]) 
    plt.imshow(unpadded_prediction, cmap=bin_cmap)
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none', zorder=3
    )
    ax.add_patch(rect)
    plt.axis('off')  # Remove axis for clean saving
    plt.savefig(os.path.join(
            pred_plots_dir,
            f'pred_for_{img_n}_for_{model_n}.png'), 
        bbox_inches='tight', dpi=300)
    plt.show()

    ###########################################################################
    '''WATERSHED'''
    pred = cv2.imread(save_path)
    pred_grey = pred[:, :, 0]
    # threshold pred to binary with 255 as class value
    ret1, thresh = cv2.threshold(
        pred_grey, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # remove small noisy predicted pixels
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        thresh, 
        cv2.MORPH_OPEN, 
        kernel, 
        iterations=2)
    # identify sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    # identify sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(
        dist_transform, 
        0.2 * dist_transform.max(), 
        255, 0)
    sure_fg = np.uint8(sure_fg)
    # determine unknown regions to process by watershed
    unknown = cv2.subtract(sure_bg, sure_fg)
    # create marker andlabel regions inside
    # sure fg and bg get positive integer label, unknown are labeled 0
    ret3, markers = cv2.connectedComponents(sure_fg)
    # avoid 0 label for background pixels which watershed treats as unknowns
    markers = markers + 10
    # change unknown pixels to 0
    markers[unknown == 255] = 0
    # apply watershed filling
    markers = cv2.watershed(pred, markers)
    # color boundaries in yellow for better visualization
    pred2 = pred.copy()
    pred2[markers == -1] = [0, 255, 255]
    pred3 = color.label2rgb(markers, bg_label=10)
    # convert black background to white and add transparency
    pred3_display = pred3.copy() 
    alpha = np.where((pred3_display == [0., 0., 0.]).all(axis=-1), 0, 0.4)
    pred3_display = np.dstack((pred3_display, alpha))

    # Save the watershed prediction
    plt.figure(figsize=(8, 8))
    plt.title(f'{img_n} Watershed {model_n}')
    plt.imshow(large_image_norm[:,:,:3])
    plt.imshow(pred3_display)
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none', zorder=3
    )
    ax.add_patch(rect)
    plt.axis('off')  # Remove axis for clean saving
    plt.savefig(os.path.join(
        watershed_plot_output_dir,
        f'watershed_{img_n}_{model_n}.png'), 
                bbox_inches='tight', dpi=300)
    plt.show()

    markers_bg_rem = np.where(markers == 10, 0, markers)
    markers_class_and_borders = np.where(markers > 0, 1, markers)
    props = measure.regionprops_table(
            markers_bg_rem,
            intensity_image=pred_grey,
            properties=["label", 
                        "area", 
                        "equivalent_diameter", 
                        "mean_intensity", 
                        "solidity"],
        )
    df = pd.DataFrame(props)
    df.to_csv(
        os.path.join(watershed_stats_output_dir, 
                    f'binprops_{img_n}_{model_n}.csv'),
        index=False,
    )    

###############################################################################
    '''
    Residual U-Net with Attention
    '''
    def get_att_res_unet (input_shape, n_classes):
        return Attention_ResUNet(input_shape, n_classes)

    model_name = "att_res_unet_bin_45ep_B16_binfocloss.hdf5"
    model_n = model_name.replace(".hdf5", "")
    att_res_unet_bin = get_att_res_unet (input_shape, n_classes)
    att_res_unet_bin.compile(
        optimizer=Adam(learning_rate = 1e-2), 
        loss=BinaryFocalLoss(gamma=2), 
        metrics=[jacard_coef])
    print(att_res_unet_bin.summary())
    att_res_unet_bin.load_weights(os.path.join(model_path, model_name))

    predicted_patches = []
    count = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:,:,:]
            single_patch_norm = single_patch / 255.0

            # Predict and threshold for values above 0.5 probability
            single_patch_prediction = (
                att_res_unet_bin.predict(
                    single_patch_norm)[0,:,:,0] > 0.5).astype(np.uint8)
            predicted_patches.append(single_patch_prediction)

    predicted_patches = np.array(predicted_patches)

    # reshape the numpy array of non-ordered patches to original image order
    predicted_patches_reshaped = np.reshape(
        predicted_patches, 
        (patches.shape[0], patches.shape[1], IMG_HEIGHT, IMG_WIDTH)
        )
    reconstructed_prediction = unpatchify(
        predicted_patches_reshaped, 
        padded_image.shape[:2])

    unpadded_prediction = reconstructed_prediction[3:-2, 3:-2]
    unpadded_prediction = 1 - unpadded_prediction

    save_path = os.path.join(
        pred_mask_output_dir, 
        f"{img_n}_pred_mask_{model_n}.tif")
    tiff.imwrite(
        save_path, 
        unpadded_prediction.astype(np.uint8))    

    # Save the image with prediction overlay
    plt.figure(figsize=(8, 8))
    plt.title(f'Prediction {img_n} & {model_n}')
    plt.imshow(large_image_norm[:, :, :3]) 
    plt.imshow(unpadded_prediction, cmap=bin_cmap)
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)
    plt.axis('off')  # Remove axis for clean saving
    plt.savefig(os.path.join(
            pred_plots_dir,
            f'pred_for_{img_n}_for_{model_n}.png'), 
        bbox_inches='tight', dpi=300)
    plt.show()

    ###########################################################################
    '''WATERSHED'''
    pred = cv2.imread(save_path)
    pred_grey = pred[:, :, 0]
    # threshold pred to binary with 255 as class value
    ret1, thresh = cv2.threshold(
        pred_grey, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # remove small noisy predicted pixels
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # identify sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    # identify sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(
        dist_transform, 
        0.2 * dist_transform.max(), 
        255, 0)
    sure_fg = np.uint8(sure_fg)
    # determine unknown regions to process by watershed
    unknown = cv2.subtract(sure_bg, sure_fg)
    # create marker andlabel regions inside
    # sure fg and bg get positive integer label, unknown are labeled 0
    ret3, markers = cv2.connectedComponents(sure_fg)
    # avoid 0 label for background pixels which watershed treats as unknowns
    markers = markers + 10
    # change unknown pixels to 0
    markers[unknown == 255] = 0
    # apply watershed filling
    markers = cv2.watershed(pred, markers)
    # color boundaries in yellow for better visualization
    pred2 = pred.copy()
    pred2[markers == -1] = [0, 255, 255]
    pred3 = color.label2rgb(markers, bg_label=10)
    # convert black background to white and add transparency
    pred3_display = pred3.copy() 
    alpha = np.where((pred3_display == [0., 0., 0.]).all(axis=-1), 0, 0.4)
    pred3_display = np.dstack((pred3_display, alpha))

    # Save the watershed prediction plot
    plt.figure(figsize=(8, 8))
    plt.title(f'{img_n} Watershed {model_n}')
    plt.imshow(large_image_norm[:,:,:3])
    plt.imshow(pred3_display)
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none', zorder=3
    )
    ax.add_patch(rect)
    plt.axis('off')  # Remove axis for clean saving
    plt.savefig(os.path.join(
        watershed_plot_output_dir,
        f'watershed_{img_n}_{model_n}.png'), 
                bbox_inches='tight', dpi=300)
    plt.show()

    markers_bg_rem = np.where(markers == 10, 0, markers)
    markers_class_and_borders = np.where(markers > 0, 1, markers)
    props = measure.regionprops_table(
            markers_bg_rem,
            intensity_image=pred_grey,
            properties=["label", 
                        "area", 
                        "equivalent_diameter", 
                        "mean_intensity", 
                        "solidity"],
        )
    df = pd.DataFrame(props)
    df.to_csv(
        os.path.join(watershed_stats_output_dir, 
                    f'binprops_{img_n}_{model_n}.csv'),
        index=False,
    )

###############################################################################
    '''multiclass predictions'''
    n_classes = 4

    padded_mask = np.pad(
    mask, 
    ((pad_top, pad_bottom), (pad_left, pad_right)), 
    mode='constant', 
    constant_values=0)

    # plot original image with existing mask part
    plt.figure(figsize=(8, 8))
    plt.title(f'Multi-class mask on {img_n}')
    plt.imshow(large_image_norm[:, :, :3]) 
    plt.imshow(padded_mask, cmap=multi_cmap)  # Overlay the padded mask
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(os.path.join(
            pred_plots_dir,
            f'mask_of_{img_n}_multi.png'), 
        bbox_inches='tight', dpi=300)
    plt.show()

###############################################################################
    '''
    Simple U-Net
    '''

    def get_multi_unet_model(n_classes):
        return multi_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model_name = "simp_unet_multi_50ep_B16_catfocCEloss.hdf5"
    model_n = model_name.replace(".hdf5", "")
    simp_unet_multi = get_multi_unet_model(n_classes)
    simp_unet_multi.load_weights(os.path.join(model_path, model_name))

    predicted_patches = []
    count = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:,:,:]
            single_patch_norm = single_patch / 255.0

            # Predict and threshold for values above 0.5 probability
            single_patch_prediction = simp_unet_multi.predict(
                single_patch_norm)
            single_patch_prediction_argmax = np.argmax(
                single_patch_prediction, axis=3)
            predicted_patches.append(single_patch_prediction_argmax)

    predicted_patches = np.array(predicted_patches)

    # reshape the numpy array of non-ordered patches to original image order
    predicted_patches_reshaped = np.reshape(
        predicted_patches, 
        (patches.shape[0], patches.shape[1], IMG_HEIGHT, IMG_WIDTH)
        )
    reconstructed_prediction = unpatchify(
        predicted_patches_reshaped, 
        padded_image.shape[:2])

    unpadded_prediction = reconstructed_prediction[3:-2, 3:-2]

    save_path = os.path.join(
        pred_mask_output_dir, 
        f"{img_n}_pred_mask_{model_n}.tif")
    tiff.imwrite(
        save_path, 
        unpadded_prediction.astype(np.uint8))

    # Save the image with prediction overlay
    plt.figure(figsize=(8, 8))
    plt.title(f'Prediction {img_n} & {model_n}')
    plt.imshow(large_image_norm[:, :, :3]) 
    plt.imshow(unpadded_prediction, cmap=multi_cmap)
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)
    plt.axis('off')  # Remove axis for clean saving
    plt.savefig(os.path.join(
            pred_plots_dir,
            f'pred_for_{img_n}_for_{model_n}.png'), 
        bbox_inches='tight', dpi=300)
    plt.show()

    ###########################################################################
    '''WATERSHED'''
    pred_single = tiff.imread(save_path)
    pred_color = cv2.imread(save_path)
    markers_final = []
    props_list = []

    for id in range(1, 4):
        pred_grey = np.where(pred_single != id, 0, 1)
        pred_grey = pred_grey.astype(np.uint8)
        pred_bin_color = np.where(
            (pred_color != [id, id, id]).all(axis=-1)[..., np.newaxis],
            [0, 0, 0],
            pred_color
        )
        pred_bin_color = pred_bin_color.astype(np.uint8)
        # pred_binary = (pred_single == 1) * 255
        ret1, thresh = cv2.threshold(
            pred_grey, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # kernel = np.ones((3, 3), np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(
            thresh, 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=10)
        dist_transform = cv2.distanceTransform(
            opening, 
            cv2.DIST_L2, 
            5)
        _, sure_fg = cv2.threshold(
            dist_transform, 
            0.2 * dist_transform.max(), 
            255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # Create markers for the watershed algorithm
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 10  # Avoid 0 label
        markers[unknown == 255] = 0
        markers = cv2.watershed(pred_bin_color, markers)
        pred2 = pred_bin_color.copy()
        pred2[markers == -1] = [0, 255, 255]
        pred3 = color.label2rgb(markers, bg_label=10)
        # convert black background to white and add transparency
        pred3_display = pred3.copy() 
        alpha = np.where((pred3_display == [0., 0., 0.]).all(axis=-1), 0, 0.4)
        pred3_display = np.dstack((pred3_display, alpha))

        # Save the watershed prediction for single class
        plt.figure(figsize=(8, 8))
        plt.title(f'{img_n} Watershed class{id} {model_n}')
        plt.imshow(large_image_norm[:,:,:3])
        plt.imshow(pred3_display)
        ax = plt.gca()
        rect = ptchs.Rectangle(
            (pad_left, pad_top),  # Bottom-left corner
            mask.shape[1],  # Width
            mask.shape[0],  # Height
            linewidth=2, edgecolor='black', facecolor='none', zorder=3
        )
        ax.add_patch(rect)
        plt.axis('off')  # Remove axis for clean saving
        plt.savefig(os.path.join(
            watershed_plot_output_dir,
            f'watershed_class{id}_{img_n}_{model_n}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.show()

        markers = np.where(markers == 10, 0, markers)
        props = measure.regionprops_table(
            markers,
            intensity_image=pred_grey,
            properties=["label", 
                        "area", 
                        "equivalent_diameter", 
                        "mean_intensity", 
                        "solidity"],
        )
        df = pd.DataFrame(props)
        df['class_id'] = id
        props_list.append(df)

        markers_class_and_borders = np.where(markers > 0, id, markers)
        markers_final.append(markers_class_and_borders)

    # save result props
    final_props_df = pd.concat(props_list, ignore_index=True)
    final_props_df.to_csv(
        os.path.join(watershed_stats_output_dir, 
                    f'multiprops_{img_n}_{model_n}.csv'),
        index=False,
    )

    # plot multiclass watershed plots
    combined_overlay = np.zeros_like(markers_final[0], dtype=int)

    for i, class_array in enumerate(markers_final, start=1):
        combined_overlay[class_array == i] = i  # Assign class label
        combined_overlay[class_array == -1] = -1  # Retain borders

    # Create an RGBA image for overlay
    rgba_overlay = np.zeros((*combined_overlay.shape, 4), dtype=float)

    # Assign colors to the RGBA overlay based on combined_overlay
    for value, col in mc_watershed_colors.items():
        pred_mask = combined_overlay == value
        rgba_overlay[pred_mask] = col

    # Plot the aerial image with overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(large_image_norm[:, :, :3])  # Plot the aerial image
    plt.imshow(rgba_overlay, interpolation='none')  # Overlay the predictions
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none', zorder=3
    ) 
    ax.add_patch(rect)
    plt.axis('off')
    plt.title(f"{img_n} Watershed {model_n}")
    plt.savefig(os.path.join(
        watershed_plot_output_dir,
        f"multi_plot_{img_n}_{model_n}.png"),
        bbox_inches='tight', dpi=300)
    plt.show()

    ###########################################################################
    '''
    Residual U-Net with attention
    '''
    def get_att_res_unet (input_shape, n_classes):
        return Attention_ResUNet(input_shape, n_classes)
    model_name = "att_res_unet_multi_40ep_B16_catfocCEloss.hdf5"
    model_n = model_name.replace(".hdf5", "")
    att_res_unet_multi = get_att_res_unet (input_shape, n_classes)
    metric = tf.keras.metrics.MeanIoU(num_classes=n_classes)
    att_res_unet_multi.compile(
        optimizer=Adam(learning_rate = 1e-2), 
        loss=keras.losses.CategoricalFocalCrossentropy(), 
        metrics=[metric])
    att_res_unet_multi.load_weights(os.path.join(model_path, model_name))

    predicted_patches = []
    count = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:,:,:]
            single_patch_norm = single_patch / 255.0

            # Predict and threshold for values above 0.5 probability
            single_patch_prediction = att_res_unet_multi.predict(
                single_patch_norm)
            single_patch_prediction_argmax = np.argmax(
                single_patch_prediction, axis=3)
            predicted_patches.append(single_patch_prediction_argmax)

    predicted_patches = np.array(predicted_patches)

    # reshape the numpy array of non-ordered patches to original image order
    predicted_patches_reshaped = np.reshape(
        predicted_patches, 
        (patches.shape[0], patches.shape[1], IMG_HEIGHT, IMG_WIDTH)
        )
    reconstructed_prediction = unpatchify(
        predicted_patches_reshaped, 
        padded_image.shape[:2])
    unpadded_prediction = reconstructed_prediction[3:-2, 3:-2]

    save_path = os.path.join(
        pred_mask_output_dir, 
        f"{img_n}_pred_mask_{model_n}.tif")
    tiff.imwrite(
        save_path, 
        unpadded_prediction.astype(np.uint8))

    # Save the image with prediction overlay
    plt.figure(figsize=(8, 8))
    plt.title(f'Prediction {img_n} & {model_n}')
    plt.imshow(large_image_norm[:, :, :3]) 
    plt.imshow(unpadded_prediction, cmap=multi_cmap)
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)
    plt.axis('off')  # Remove axis for clean saving
    plt.savefig(os.path.join(
            pred_plots_dir,
            f'pred_for_{img_n}_for_{model_n}.png'), 
        bbox_inches='tight', dpi=300)
    plt.show()

    ###########################################################################
    '''WATERSHED'''
    pred_single = tiff.imread(save_path)
    pred_color = cv2.imread(save_path)
    markers_final = []
    props_list = []

    for id in range(1, 4):
        pred_grey = np.where(pred_single != id, 0, 1)
        pred_grey = pred_grey.astype(np.uint8)
        pred_bin_color = np.where(
            (pred_color != [id, id, id]).all(axis=-1)[..., np.newaxis],
            [0, 0, 0],
            pred_color
        )
        pred_bin_color = pred_bin_color.astype(np.uint8)
        # pred_binary = (pred_single == 1) * 255
        ret1, thresh = cv2.threshold(
            pred_grey, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(
            thresh, 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=10)
        dist_transform = cv2.distanceTransform(
            opening, 
            cv2.DIST_L2, 
            5)
        _, sure_fg = cv2.threshold(
            dist_transform, 
            0.2 * dist_transform.max(), 
            255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # Create markers for the watershed algorithm
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 10  # Avoid 0 label
        markers[unknown == 255] = 0
        markers = cv2.watershed(pred_bin_color, markers)
        pred2 = pred_bin_color.copy()
        pred2[markers == -1] = [0, 255, 255]
        pred3 = color.label2rgb(markers, bg_label=10)
        # convert black background to white and add transparency
        pred3_display = pred3.copy() 
        alpha = np.where((pred3_display == [0., 0., 0.]).all(axis=-1), 0, 0.4)
        pred3_display = np.dstack((pred3_display, alpha))

        # Save the watershed prediction for single class
        plt.figure(figsize=(8, 8))
        plt.title(f'{img_n} Watershed class{id} {model_n}')
        plt.imshow(large_image_norm[:,:,:3])
        plt.imshow(pred3_display)
        ax = plt.gca()
        rect = ptchs.Rectangle(
            (pad_left, pad_top),  # Bottom-left corner
            mask.shape[1],  # Width
            mask.shape[0],  # Height
            linewidth=2, edgecolor='black', facecolor='none', zorder=3
        )
        ax.add_patch(rect)
        plt.axis('off')  # Remove axis for clean saving
        plt.savefig(os.path.join(
            watershed_plot_output_dir,
            f'watershed_class{id}_{img_n}_{model_n}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.show()    

        markers = np.where(markers == 10, 0, markers)
        props = measure.regionprops_table(
            markers,
            intensity_image=pred_grey,
            properties=["label", 
                        "area", 
                        "equivalent_diameter", 
                        "mean_intensity", 
                        "solidity"],
        )
        df = pd.DataFrame(props)
        df['class_id'] = id
        props_list.append(df)

        markers_class_and_borders = np.where(markers > 0, id, markers)
        markers_final.append(markers_class_and_borders)

    # save result props
    final_props_df = pd.concat(props_list, ignore_index=True)
    final_props_df.to_csv(
        os.path.join(watershed_stats_output_dir, 
                    f'multiprops_{img_n}_{model_n}.csv'),
        index=False,
    )

    # plot multiclass watershed plots
    combined_overlay = np.zeros_like(markers_final[0], dtype=int)

    for i, class_array in enumerate(markers_final, start=1):
        combined_overlay[class_array == i] = i  # Assign class label
        combined_overlay[class_array == -1] = -1  # Retain borders

    # Create an RGBA image for overlay
    rgba_overlay = np.zeros((*combined_overlay.shape, 4), dtype=float)

    # Assign colors to the RGBA overlay based on combined_overlay
    for value, col in mc_watershed_colors.items():
        pred_mask = combined_overlay == value
        rgba_overlay[pred_mask] = col

    # Plot the aerial image with overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(large_image_norm[:, :, :3])  # Plot the aerial image
    plt.imshow(rgba_overlay, interpolation='none')  # Overlay the predictions
    ax = plt.gca()
    rect = ptchs.Rectangle(
        (pad_left, pad_top),  # Bottom-left corner
        mask.shape[1],  # Width
        mask.shape[0],  # Height
        linewidth=2, edgecolor='black', facecolor='none', zorder=3
    )
    ax.add_patch(rect)
    plt.axis('off')
    plt.title(f"{img_n} Watershed {model_n}")
    plt.savefig(os.path.join(
        watershed_plot_output_dir,
        f"multi_plot_{img_n}_{model_n}.png"),
        bbox_inches='tight', dpi=300)
    plt.show()


###############################################################################
'''06 Statistical analysis of watershed statistics'''
###############################################################################

# Define the base directory and file naming conventions
data_dir = "/data_hdd/pauline/dataset/swf/watershed_res/"
image_ids = ["AEG27", "HEW16", "SEG29", "SEG38", "SEG42", "SEW08"]

# Define the models and their corresponding file patterns
models = {
    "binary_att_res_unet": "binprops_{image_id}_att_res_unet_bin_45ep_B16_binfocloss.csv",
    "binary_simp_unet": "binprops_{image_id}_simp_unet_bin_50ep_B16_binCEloss.csv",
    "multi_att_res_unet": "multiprops_{image_id}_att_res_unet_multi_40ep_B16_catfocCEloss.csv",
    "multi_simp_unet": "multiprops_{image_id}_simp_unet_multi_50ep_B16_catfocCEloss.csv",
}

# Pixel size in meters
pixel_size = 0.02

# Load and concatenate dataframes
model_data = {}
for model_name, file_pattern in models.items():
    model_data[model_name] = []
    for image_id in image_ids:
        file_path = os.path.join(
            data_dir, 
            file_pattern.format(image_id=image_id))
        df = pd.read_csv(file_path)
        df['image_id'] = image_id  # Assign image ID
        df['model'] = model_name  # Assign model name
        df['area_m2'] = df['area'] * (pixel_size ** 2)  # Calculate area in m^2
        model_data[model_name].append(df)
    model_data[model_name] = pd.concat(
        model_data[model_name], 
        ignore_index=True)

# Combine all models into one dataframe for comparison
combined_data = pd.concat(model_data.values(), ignore_index=True)

# Define helper functions for analysis
def summarize_metrics(df, group_by='image_id'):
    """Summarize metrics for a given dataframe grouped by a column."""
    summary = df.groupby(group_by).agg({
        'area': ['mean', 'median', 'std'],
        'area_m2': ['mean', 'median', 'std'],
        'equivalent_diameter': ['mean', 'median', 'std'],
        'solidity': ['mean'],
        'label': 'count'
    })
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.rename(columns={'label_count': 'instance_count'})
    return summary

def summarize_class_instances(df):
    """Sum. inst. count & area p. image & p. cl. of multi-class segm."""
    if 'class_id' in df.columns:
        class_summary = df.groupby(['image_id', 'class_id']).agg({
            'area_m2': ['sum', 'mean'],
            'label': 'count'
        })
        class_summary.columns = [
            'total_area_m2', 
            'mean_area_m2', 
            'instance_count']
        class_summary = class_summary.reset_index()

        # Pivot the data to create a clear summary table
        summary_pivot = class_summary.pivot(index='image_id', 
                                            columns='class_id')
        summary_pivot.columns = [
            f"class_{int(col[1])}_{col[0]}" for col in summary_pivot.columns]
        summary_pivot['total_area_m2'] = summary_pivot.filter(
            like='total_area_m2').sum(axis=1)
        summary_pivot['total_instances'] = summary_pivot.filter(
            like='instance_count').sum(axis=1)
        return summary_pivot
    else:
        return None
    
def save_plot(output_dir, filename):
    """Save the current matplotlib plot to the specified directory."""
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()  # Close the plot to free memory

def plot_instance_count(df, title, save_path):
    """Plot instance counts per image."""
    counts = df.groupby('image_id').size()
    counts.plot(kind='bar', title=title)
    plt.xlabel('Image ID')
    plt.ylabel('Instance Count')
    save_plot(save_path, 
              f"{title.replace(' ', '_').lower()}_instance_count.png")

def plot_class_distribution(df, title, save_path):
    """Plot class distribution for multi-class segmentation."""
    if 'class_id' in df.columns:
        class_counts = df.groupby(
            ['image_id', 'class_id']).size().unstack(fill_value=0)
        class_counts = class_counts.reindex(
            columns=[1, 2, 3], fill_value=0)  # Ensure all classes are present
        class_counts.plot(kind='bar', 
                          stacked=True, 
                          title=title, 
                          color=['red', 'green', 'blue'])
        plt.xlabel('Image ID')
        plt.ylabel('Instance Count')
        plt.legend(title='Class ID')
        save_plot(save_path, 
                  f"{title.replace(' ', '_').lower()}_class_distribution.png")

def plot_model_comparison(df, title, save_path):
    """Plot comparison of instance counts across models for each image."""
    counts = df.groupby(['image_id', 'model']).size().unstack(fill_value=0)
    counts.plot(kind='bar', stacked=False, title=title)
    plt.xlabel('Image ID')
    plt.ylabel('Instance Count')
    plt.legend(title='Model')
    save_plot(save_path, 
              f"{title.replace(' ', '_').lower()}_model_comparison.png")

def plot_area_model_comparison(df, title, save_path):
    """Plot comparison of predicted area (m²) across models for each image."""
    area_sums = df.groupby(
        ['image_id', 'model']).agg({'area_m2': 'sum'}).unstack(fill_value=0)
    ax = area_sums.plot(kind='bar', stacked=False, title=title)
    plt.xlabel('Image ID')
    plt.ylabel('Total Area (m²)')
    plt.legend(labels=['binary_att_res_unet', 
                       'binary_simp_unet', 
                       'multi_att_res_unet', 
                       'multi_simp_unet'], 
                       title='Model')
    save_plot(save_path, 
              f"{title.replace(' ', '_').lower()}_area_comparison.png")

def plot_model_class_comparison(df, title, save_path):
    """Plot comparison of instance counts by class for each image and model."""
    if 'class_id' in df.columns:
        grouped = df.groupby(
            ['image_id', 
             'model', 
             'class_id']).agg({'area_m2': 'sum'}).unstack(fill_value=0)
        for image_id in grouped.index.get_level_values(0).unique():
            image_data = grouped.loc[image_id]
            existing_classes = image_data.columns.get_level_values(1).unique()
            colors = ['red', 'green', 'blue'][:len(existing_classes)]
            ax = image_data.plot(kind='bar', stacked=True, 
                                 title=f"{title} - {image_id}", color=colors)
            ax.set_xlabel('Model')
            ax.set_ylabel('Area (m²)')
            plt.xticks(rotation=0)  # Set x-axis labels to horizontal
            plt.legend(labels=[str(int(cls)) for cls in existing_classes], 
                       title='Class ID')
            save_plot(
                save_path, 
                f"{title.replace(' ', '_').lower()}_instcount_comparison.png")

def remove_outliers(df, column, group_by):
    """Remove outliers based on the IQR method."""
    def filter_outliers(group):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return group[(
            group[column] >= lower_bound) & (
                group[column] <= upper_bound)]
    
    return df.groupby(group_by, group_keys=False).apply(filter_outliers)

def plot_mean_instance_area_boxplots(df):
    """Boxplots of mean inst. area (m²) of each model across all imgs & images."""
    # Remove outliers for boxplots
    filtered_df = remove_outliers(df, column='area_m2', group_by='model')

    # Overall boxplot for all models
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=filtered_df, x='model', y='area_m2')
    plt.title(
        "Boxplot of Mean Instance Area (m²) Across All Images (Without Outliers)")
    plt.xlabel('Model')
    plt.ylabel('Mean Instance Area (m²)')
    # Save the plot with a meaningful name
    overall_filename = "boxplot_mean_instance_area_all_models.png"
    plt.savefig(os.path.join(watershed_stats_output_dir, 
                             overall_filename))
    plt.close()  # Close the plot to free memory

    # Per-image boxplot for each model
    for image_id in df['image_id'].unique():
        filtered_image_df = remove_outliers(
            df[df['image_id'] == image_id], column='area_m2', group_by='model')
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=filtered_image_df, x='model', y='area_m2')
        plt.title(
            f"Boxplot of Mean Instance Area (m²) for Each Model - Image {image_id} (Without Outliers)")
        plt.xlabel('Model')
        plt.ylabel('Mean Instance Area (m²)')
         # Save the plot with a meaningful name for each image
        individual_filename = f"boxplot_mean_instance_area_{image_id}.png"
        plt.savefig(os.path.join(watershed_stats_output_dir, 
                                 individual_filename))
        plt.close()  # Close the plot to free memory

def plot_solidity_boxplots(df):
    """Plot boxplots of solidity for each model and for each image."""
    # Overall boxplot for each model
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='model', y='solidity')
    plt.title("Boxplot of Solidity Across All Images")
    plt.xlabel('Model')
    plt.ylabel('Solidity')
    # Save the plot with a meaningful name
    overall_filename = "boxplot_solidity_all_models.png"
    plt.savefig(os.path.join(watershed_stats_output_dir, 
                             overall_filename))
    plt.close()  # Close the plot to free memory

    # Per-image boxplot for each model
    for image_id in df['image_id'].unique():
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df[df['image_id'] == image_id], x='model', y='solidity')
        plt.title(f"Boxplot of Solidity for Each Model - Image {image_id}")
        plt.xlabel('Model')
        plt.ylabel('Solidity')
         # Save the plot with a meaningful name for each image
        individual_filename = f"boxplot_solidity_{image_id}.png"
        plt.savefig(os.path.join(watershed_stats_output_dir, 
                                 individual_filename))
        plt.close()  # Close the plot to free memory

# Perform analyses and generate outputs
for model_name, df in model_data.items():
    print(f"Analysis for {model_name}:")

    # Summarize metrics
    summary = summarize_metrics(df)
    print(summary)

    # Save summary to CSV
    summary.to_csv(os.path.join(
        watershed_stats_output_dir,
        f"{model_name}_watershed_summary.csv"))

    # Plot instance counts
    plot_instance_count(df, 
                        title=f"Instance Count per Image ({model_name})", 
                        save_path=watershed_stats_output_dir)

    # Summarize and save class distribution for multi-class models
    if 'class_id' in df.columns:
        class_summary = summarize_class_instances(df)
        if class_summary is not None:
            print(class_summary)
            class_summary.to_csv(os.path.join(
                watershed_stats_output_dir,
                f"{model_name}_watershed_class_summary.csv"))

        # Plot class distributions
        plot_class_distribution(df, 
                                title=f"Class Distribution ({model_name})", 
                                save_path=watershed_stats_output_dir)

# Plot overall comparison across models
plot_model_comparison(
    combined_data, 
    title="Model Comparison: Instance Counts Across Images",
    save_path=watershed_stats_output_dir)
plot_area_model_comparison(
    combined_data, 
    title="Model Comparison: Total Area (m²) Across Images",
    save_path=watershed_stats_output_dir)
plot_model_class_comparison(
    combined_data, 
    title="Model Comparison: Class-Specific Area (m²)",
    save_path=watershed_stats_output_dir)
plot_mean_instance_area_boxplots(combined_data)
plot_solidity_boxplots(combined_data)