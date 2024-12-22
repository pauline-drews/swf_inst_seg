'''
05 Model Evaluation

This is the 5th step of the workflow for my Master thesis with the title: 

"Instance-Aware Semantic Segmentation of Small Woody Landscape Features 
 Using High-Resolution Aerial Images in Biodiversity Exploratories"

Parts of this script were taken and modified from the tutorial scripts of
Dr. Sreenivas Bhattiprolu:
https://github.com/bnsreenu/python_for_microscopists/blob/master/204_train_simple_unet_for_mitochondria.py
https://github.com/bnsreenu/python_for_microscopists/blob/master/208_multiclass_Unet_sandstone.py
https://github.com/bnsreenu/python_for_microscopists/blob/master/224_225_226_mito_segm_using_various_unet_models.py 

This script follows after script 04_train_models.py and is followed 
by script 06_model_prediction_and_watershed.py.

In script 04_train_models.py, a simple U-Net and a residual U-Net with
attention were trained for binary and multi-class case. Training stats (loss, 
metrics) were plotted and the models were saved.

In this script, the models are tested on testing data and performance can thus 
be quantified by calculating IoU metric for binary case and mean IoU & per
class IoU for multi-class case.

In script 06_model_prediction_and_watershed, the final models will be used to
predict small woody features on some example large images. This prediction will
also be made instance-aware by applying watershed algorithm. Plots of 
predictions, watershed results and the overlay of the original mask are created
and the stats about the individual instances will be extracted and saved.
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
from keras.metrics import MeanIoU
import csv

# General parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 4
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model_path = "/data_hdd/pauline/swf/results/model_weights/b_final_models/"
base_path = "/data_hdd/pauline/swf/dataset/patches_256x256/split_datasets/"
paths = {
    "binary": os.path.join(base_path, "binary/"),
    # "class1": os.path.join(base_path, "class1/"),
    # "class2": os.path.join(base_path, "class2/"),
    # "class3": os.path.join(base_path, "class3/"),
    # "trees_hedges": os.path.join(base_path, "trees_hedges/"),
    "multiclass": os.path.join(base_path, "multiclass/")
}
model_evaluation_dir = "/data_hdd/pauline/swf/results/model_performance/"
os.makedirs(model_evaluation_dir, exist_ok=True)
csv_file_path = os.path.join(model_evaluation_dir, "results_metrics.csv")

def load_data(path, split):
    images_path = os.path.join(path, split, "images/")
    masks_path = os.path.join(path, split, "masks/")
    X, y = [], []

    for filename in os.listdir(images_path):
        if filename.endswith('.tif'):  # Ensure only TIFF files are loaded
            img_path = os.path.join(images_path, filename)
            img = tiff.imread(img_path)  # Read the image as a NumPy array
            X.append(img)

            # Construct mask filename by adding '_mask' before the extension
            mask_filename = filename[:-4] + '_mask.tif'  # Remove .tif and add _mask.tif
            mask_path = os.path.join(masks_path, mask_filename)
            mask = tiff.imread(mask_path)  # Read the mask as a NumPy array
            y.append(mask)

    return np.array(X), np.array(y)

# Function to save metrics to CSV
def save_metrics_to_csv(model_name, metrics):
    # Check if the file exists
    file_exists = os.path.isfile(csv_file_path)

    # Open the file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file doesn't exist
        if not file_exists:
            writer.writerow(["Model Name", "Metric", "Value"])

        # Write metrics
        for metric, value in metrics.items():
            writer.writerow([model_name, metric, value])
###############################################################################
# a) binary
X_test, y_test = load_data(paths["binary"], "test")
n_classes = len(np.unique(y_test))

'''
Simple U-Net
'''
def get_simple_unet_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model_name = "simp_unet_bin_50ep_B16_binCEloss.hdf5"
simp_unet_bin = get_simple_unet_model()
simp_unet_bin.load_weights(os.path.join(model_path, model_name))

y_pred=simp_unet_bin.predict(X_test)
y_pred_thresholded = y_pred > 0.05

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)
save_metrics_to_csv(model_name, {"IoU": iou_score})

'''
Residual U-Net with attention
'''
def get_att_res_unet (input_shape, n_classes):
    return Attention_ResUNet(input_shape, n_classes)

model_name = "att_res_unet_bin_45ep_B16_binfocloss.hdf5"
att_res_unet_bin = get_att_res_unet (input_shape, n_classes)
att_res_unet_bin.compile(
    optimizer=Adam(learning_rate = 1e-2), 
    loss=BinaryFocalLoss(gamma=2), 
    metrics=[jacard_coef])
print(att_res_unet_bin.summary)
att_res_unet_bin.load_weights(os.path.join(model_path, model_name))

y_pred=att_res_unet_bin.predict(X_test)
y_pred_thresholded = y_pred > 0.05

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)
save_metrics_to_csv(model_name, {"IoU": iou_score})

###############################################################################
# f) multiclass
X_test, y_test = load_data(paths["multiclass"], "test")
n_classes = len(np.unique(y_test))

'''
Simple U-Net
'''
def get_multi_unet_model(n_classes):
    return multi_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model_name = "simp_unet_multi_50ep_B16_catfocCEloss.hdf5"
simp_unet_multi = get_multi_unet_model(n_classes)
simp_unet_multi.load_weights(os.path.join(model_path, model_name))

y_pred=simp_unet_multi.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

# IoU
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test, y_pred_argmax)
mean_iou = IOU_keras.result().numpy()
print("Mean IoU =", mean_iou)

# IoU for each class
confusion_matrix = IOU_keras.total_cm.numpy()  # total_cm stores the confusion matrix
values = confusion_matrix.reshape(n_classes, n_classes)
class_ious = []
for i in range(n_classes):
    intersection = values[i, i]
    union = (
        np.sum(values[i, :]) +  # Sum of the row (True Positives + False Negatives)
        np.sum(values[:, i]) -  # Sum of the column (True Positives + False Positives)
        values[i, i]            # Subtract the intersection (True Positives)
    )
    iou = intersection / union if union != 0 else 0
    class_ious.append(iou)

print("IoU for each class:", class_ious)
metrics = {"Mean IoU": mean_iou}
for idx, iou in enumerate(class_ious):
    metrics[f"Class {idx} IoU"] = iou
save_metrics_to_csv(model_name, metrics)

'''
Residual U-Net with attention
'''
model_name = "att_res_unet_multi_40ep_B16_catfocCEloss.hdf5"
att_res_unet_multi = get_att_res_unet (input_shape, n_classes)
metric = tf.keras.metrics.MeanIoU(num_classes=n_classes)
att_res_unet_multi.compile(
    optimizer=Adam(learning_rate = 1e-2), 
    loss=keras.losses.CategoricalFocalCrossentropy(), 
    metrics=[metric])
att_res_unet_multi.load_weights(os.path.join(model_path, model_name))

y_pred=att_res_unet_multi.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

# IoU
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test, y_pred_argmax)
mean_iou = IOU_keras.result().numpy()
print("Mean IoU =", mean_iou)

# IoU for each class
confusion_matrix = IOU_keras.total_cm.numpy()  # total_cm stores the confusion matrix
values = confusion_matrix.reshape(n_classes, n_classes)
class_ious = []
for i in range(n_classes):
    intersection = values[i, i]
    union = (
        np.sum(values[i, :]) +  # Sum of the row (True Positives + False Negatives)
        np.sum(values[:, i]) -  # Sum of the column (True Positives + False Positives)
        values[i, i]            # Subtract the intersection (True Positives)
    )
    iou = intersection / union if union != 0 else 0
    class_ious.append(iou)

print("IoU for each class:", class_ious)
metrics = {"Mean IoU": mean_iou}
for idx, iou in enumerate(class_ious):
    metrics[f"Class {idx} IoU"] = iou
save_metrics_to_csv(model_name, metrics)