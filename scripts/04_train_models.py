"""
04 Model training

This is the 4th step of the workflow for my Master thesis with the title: 

"Instance-Aware Semantic Segmentation of Small Woody Landscape Features 
 Using High-Resolution Aerial Images in Biodiversity Exploratories"

Large parts of this script were taken and modified from the tutorial scripts of
Dr. Sreenivas Bhattiprolu:
https://github.com/bnsreenu/python_for_microscopists/blob/master/204_train_simple_unet_for_mitochondria.py
https://github.com/bnsreenu/python_for_microscopists/blob/master/208_multiclass_Unet_sandstone.py
https://github.com/bnsreenu/python_for_microscopists/blob/master/224_225_226_mito_segm_using_various_unet_models.py 

This script follows after script 03_dataset_preparation.py and is followed 
by script 05_model_evaluation.py.

In script 03_dataset_preparation.py, a dataset of 5627 256x256 rgb-nir image
patches and the corresponding masks was prepared and split into datasets of
different mask variants (binary, each individual class, trees vs. hedges, and
multiclass). Each dataset was split into train validation and test set and
saved to /data_hdd/pauline/dataset/swf/split_datasets/.

In this script, a simple U-Net model and a residual U-Net model with attention
are trained for binary and multi-class case. Training for the other mask
variants created in script 03_data_preparation is theoretically possible but 
was not performed for this thesis due to insufficient temporal resources. 
After training, training stats (loss, metrics) are plotted and the models are 
saved.

In script 05_model_evaluation, the models will be tested on testing data to
calculate final performance metrics.
"""
###############################################################################
# Load data
###############################################################################
'''
This script is run using segmod environment. The corresponding 
'''

import os
import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simple_unet_model import simple_unet_model
from simple_multi_unet_model import multi_unet_model
import segmentation_models as sm
from unet.scripts.res_unet_attention import Attention_ResUNet, jacard_coef
from focal_loss import BinaryFocalLoss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from datetime import datetime 
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Conv2D
# from sklearn.preprocessing import MinMaxScaler

# Paths
data_path = "/data_hdd/pauline/swf/dataset/patches_256x256/split_datasets/"
paths = {
    "binary": os.path.join(data_path, "binary/"),
    # "class1": os.path.join(data_path, "class1/"),
    # "class2": os.path.join(data_path, "class2/"),
    # "class3": os.path.join(data_path, "class3/"),
    # "trees_hedges": os.path.join(data_path, "trees_hedges/"),
    "multiclass": os.path.join(data_path, "multiclass/")
}
stat_output_dir_test_fits = "/data_hdd/pauline/swf/training_stats$ cd a_test_runs/"
stat_output_dir_final_fits = "/data_hdd/pauline/swf/training_stats/b_final_models/"
model_output_dir_test_fits = "/data_hdd/pauline/swf/results/model_weights/test_runs/"
model_output_dir_final_fits = "/data_hdd/pauline/swf/results/model_weights/b_final_models/"

def load_data(path, split):
    images_path = os.path.join(path, split, "images/")
    masks_path = os.path.join(path, split, "masks/")
    X, y = [], []

    for filename in os.listdir(images_path):
        if filename.endswith('.tif'):
            img_path = os.path.join(images_path, filename)
            img = tiff.imread(img_path)
            X.append(img)
            mask_filename = filename[:-4] + '_mask.tif'
            mask_path = os.path.join(masks_path, mask_filename)
            mask = tiff.imread(mask_path)
            y.append(mask)

    return np.array(X), np.array(y)

###############################################################################
# Model parameters
###############################################################################
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 4
batch_size = 16
epochs = 50
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

###############################################################################
# Data logging
###############################################################################
# Function to log execution time in minutes
def log_execution_time(model_name, execution_time, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, f"{model_name}_exetime.txt")
    
    # Convert execution_time to minutes
    execution_time_in_minutes = execution_time.total_seconds() / 60
    
    # Write execution time to file
    with open(file_path, "w") as file:
        file.write(f"Model: {model_name}\n")
        file.write(
            f"Execution Time: {execution_time_in_minutes:.2f} minutes\n")

# Function to save model training histories to csv for plotting
def save_hist_as_csv(hist, output_file):
    pd.DataFrame(hist.history).to_csv(output_file, index=False)
    print(f"History saved to {output_file}")

###############################################################################
# Define models
###############################################################################
'''
Simple U-Net single class
'''
def get_simple_unet_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

'''
Simple U-Net multi class
'''
def get_multi_unet_model(n_classes):
    return multi_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


'''
Residual U-Net with attention
'''
def get_att_res_unet (input_shape, n_classes):
    return Attention_ResUNet(input_shape, n_classes)

###############################################################################
# Early Stopping
###############################################################################
# Add smoothed early stopping mechanism to stop model when not improving
# anymore but keep best weights until then
class SmoothEarlyStopping(Callback):
    def __init__(self, 
                 monitor='val_loss', 
                 patience=10, 
                 min_delta=0.001, 
                 restore_best_weights=True):
        super(SmoothEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best = np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        # Check if validation loss decreased
        if current < self.best - self.min_delta:
            self.best = current
            self.best_weights = self.model.get_weights()
            self.wait = 0  # Reset patience counter
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.restore_best_weights:
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True


###############################################################################
# Train models
###############################################################################

# a) binary
X_train, y_train = load_data(paths["binary"], "train")
X_val, y_val = load_data(paths["binary"], "val")
n_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train) # one-hot encoded masks
y_val_cat = to_categorical(y_val)
activation = "sigmoid"

'''
Simple U-Net
'''
# adapt epochs and batch size to case and memory
epochs = 50
# epochs = 100
batch_size = 16

loss_n = "binCE"
metr_n = "binIoU"

simp_unet_bin = get_simple_unet_model()
smooth_early_stopping = SmoothEarlyStopping( # only for final model run
    monitor='val_loss', 
    patience=10, # adapt such that model stops early enough but not too early
    min_delta=0.025) # adapt -,,-
print("Simple U-Net for binary data running.")
start = datetime.now()
hist_simp_unet_bin = simp_unet_bin.fit(
    X_train, y_train, 
    batch_size = batch_size, 
    verbose=1, 
    epochs=epochs, 
    validation_data=(X_val, y_val), 
    shuffle=True,
    callbacks = [smooth_early_stopping] # only for final model run
    )
stop = datetime.now()
print("Simple U-Net for binary data finished.")
execution_time = stop-start
print("Execution time of simple U-Net for binary data: ", execution_time)
log_execution_time(
    f"simp_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss", 
    execution_time, # choose path for current case
    # stat_output_dir_test_fits,
    stat_output_dir_final_fits,
    )
simp_unet_bin.save(
    os.path.join( # choose path for current case
        # model_output_dir_test_fits,
        model_output_dir_final_fits,
        f"simp_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss.hdf5"
    )
)
save_hist_as_csv(
    hist_simp_unet_bin,
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits, 
        stat_output_dir_final_fits,
        f"simp_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss_hist.csv"
    )
)

# loss and metric plots
hist_dir = os.path.join( # choose path for current case
    # stat_output_dir_test_fits,
    stat_output_dir_final_fits,
    f"simp_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss_hist.csv"
)
data = pd.read_csv(hist_dir)

# Plotting performance metric
plt.figure(figsize=(10, 5))
plt.plot(data.index, 
         data['BinaryIoU'], 
         label='Binary IoU')
plt.plot(data.index, 
         data['val_BinaryIoU'], 
         label='Validation Binary IoU', 
         linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Binary IoU')
# choose title for current case
# plt.title('Binary simple U-Net: training and validation metrics of test run')
plt.title('Binary simple U-Net: training and validation metrics of final run')
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"simp_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss_{metr_n}_plot.png"),
    dpi=300)
plt.show()

# Plotting loss
plt.figure(figsize=(10, 5))
plt.plot(data.index, 
         data['loss'], 
         label='Loss')
plt.plot(data.index, 
         data['val_loss'], 
         label='Validation Loss', 
         linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# choose title for current case
# plt.title('Binary simple U-Net: training and validation loss of test run')
plt.title('Binary simple U-Net: training and validation loss of final run')
plt.legend()
plt.grid(True)
plt.savefig( # choose path for correct case
    os.path.join(
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"simp_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss_plot.png"),    
    dpi=300)
plt.show()


'''
Residual U-Net with attention
'''

# adapt epochs and batch size to case and memory
epochs = 50
# epochs = 100
batch_size = 16

loss_n = "binfoc"
metr_n = "binIoU"

att_res_unet_bin = get_att_res_unet (input_shape, n_classes)
att_res_unet_bin.compile(
    optimizer=Adam(learning_rate = 1e-2), 
    loss=BinaryFocalLoss(gamma=2), 
    metrics=[jacard_coef])
print(att_res_unet_bin.summary())
print("Residual U-Net with attention for binary data running.")
start = datetime.now() 
hist_att_res_unet_bin = att_res_unet_bin.fit(
    X_train, y_train_cat, 
    verbose=1,
    batch_size = batch_size,
    validation_data=(X_val, y_val_cat), 
    shuffle=True,
    epochs=epochs)
stop = datetime.now()
print("Residual U-Net with attention for binary data finished.")
execution_time = stop-start
print(
    "Execution time for Residual U-Net with attention for binary data:", 
    execution_time)
log_execution_time(
    f"att_res_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss", 
    execution_time, # choose path for current case
    # stat_output_dir_test_fits, 
    stat_output_dir_final_fits
)
att_res_unet_bin.save(
    os.path.join( # choose path for current case
        # model_output_dir_test_fits,
        model_output_dir_final_fits,
        f".att_res_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss.hdf5"
    )
)
save_hist_as_csv(
    hist_att_res_unet_bin, 
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"att_res_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss_hist.csv"
    )
)

# Loss and metric plot
csv_dir = os.path.join( # choose path for current case
    # stat_output_dir_test_fits,
    stat_output_dir_final_fits,
    f"att_res_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss_hist.csv"
)
data = pd.read_csv(csv_dir)

# metric plot
plt.figure(figsize=(10, 5))
plt.plot(data.index, 
         data['jacard_coef'], 
         label='IoU')
plt.plot(data.index, 
         data['val_jacard_coef'], 
         label='Validation IoU', 
         linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('IoU')
# choose title for current case
# plt.title('Binary residual U-net with attention: training and validation metric of test run')
plt.title('Binary residual U-net with attention: training and validation metric of final run')
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"att_res_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss_{metr_n}_plot.png"), 
    dpi=300)
plt.show()

# loss plot
plt.figure(figsize=(10, 5))
plt.plot(data.index, 
         data['loss'], 
         label='Loss')
plt.plot(data.index, 
         data['val_loss'], 
         label='Validation Loss', 
         linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# Choose title for current case
# plt.title('Binary residual U-net with attention: training and validation loss of test run')
plt.title('Binary residual U-net with attention: training and validation loss of final run')
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"att_res_unet_bin_{epochs}ep_B{batch_size}_{loss_n}loss_plot.png"), 
    dpi=300)
plt.show()

###############################################################################
# it could be added model trainings for the dataset of the other mask variants
###############################################################################

###############################################################################
# f) multiclass
X_train, y_train = load_data(paths["multiclass"], "train")
X_val, y_val = load_data(paths["multiclass"], "val")
print("Preprocessing multiclass data for backbone model")
X_train_pp = preprocess_input(X_train)
X_val_pp = preprocess_input(X_val)
n_classes = len(np.unique(y_train))
activation = "softmax"
y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_val_cat = to_categorical(y_val, num_classes=n_classes)


'''
Simple U-Net
'''

# adapt epochs and batch size to case and memory
epochs = 50
# epochs = 100
batch_size = 16

loss_n = "catfocCE"
metr_n = "mIoU"

simp_unet_multi = get_multi_unet_model(n_classes)
print("Simple U-Net for multi-class data running.")
start = datetime.now()
hist_simp_unet_multi = simp_unet_multi.fit(
    X_train, y_train_cat, 
    batch_size = batch_size, 
    verbose=1, 
    epochs=epochs, 
    validation_data=(X_val, y_val_cat), 
    shuffle=True)
stop = datetime.now()
print("Simple U-Net for multi-class data finished.")
execution_time = stop-start
print("Execution time of simple U-Net for multi-class data:", execution_time)
log_execution_time(
    f"simp_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss", 
    execution_time, # choose path for current case
    # stat_output_dir_test_fits, 
    stat_output_dir_final_fits
)
simp_unet_multi.save(
    os.path.join( # choose path for current case
        # model_output_dir_test_fits,
        model_output_dir_final_fits,
        f"simp_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss.hdf5"
    )
)
save_hist_as_csv(
    hist_simp_unet_multi, 
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"simp_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss_hist.csv"
    )
)

csv_dir = os.path.join( # choose path for current case
    # stat_output_dir_test_fits,
    stat_output_dir_final_fits,
    f"simp_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss_hist.csv")
data = pd.read_csv(csv_dir)

# Plotting Jacard Coefficient
plt.figure(figsize=(10, 5))
plt.plot(data.index, 
         data['mean_io_u'], 
         label='Mean IoU')
plt.plot(data.index, 
         data['val_mean_io_u'], 
         label='Validation mean IoU', 
         linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Mean IoU')
# Choose title for current case
# plt.title('Multi-class simple U-net: training and validation metric of test run')
plt.title('Multi-class simple U-net: training and validation metric of final run')
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"simp_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss_{metr_n}_plot.png"),
    dpi=300)
plt.show()

# Plotting Loss
plt.figure(figsize=(10, 5))
plt.plot(data.index, 
         data['loss'], 
         label='Loss')
plt.plot(data.index, 
         data['val_loss'], 
         label='Validation Loss', 
         linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# Choose title for current case
# plt.title('Multi-class simple U-net: training and validation loss of test run')
plt.title('Multi-class simple U-net: training and validation loss of final run')
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"simp_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss_plot.png"),
    dpi=300)
plt.show()


'''
U-Net with Resnet34 Backbone and Imagenet weights
'''

# adapt epochs and batch size to case and memory
epochs = 50
# epochs = 100
batch_size = 16

loss_n = "catfocCE"
metr_n = "mIoU"

unet_resnet_imagenet_multi = get_unet_resnet_imagenet(
    n_classes, activation)
new_input = Input(shape=(256, 256, 4))
x = Conv2D(3, (1, 1), activation='linear', padding='same')(new_input)
output = unet_resnet_imagenet_multi(x)
unet_resnet_imagenet_multi = Model(inputs=new_input, outputs=output)
metric = tf.keras.metrics.MeanIoU(num_classes=n_classes)
unet_resnet_imagenet_multi.compile(
    'Adam', 
    loss=tf.keras.losses.CategoricalFocalCrossentropy(), 
    metrics=[metric])
print(unet_resnet_imagenet_multi.summary())
print("Resnet34 U-Net with Imagenet weights for multi-class data running.")
start = datetime.now()
hist_unet_resnet_imagenet_multi=unet_resnet_imagenet_multi.fit(
    X_train_pp, 
    y_train_cat,
    batch_size=batch_size, 
    epochs=epochs,
    verbose=1,
    validation_data=(X_val_pp, y_val_cat))
stop = datetime.now()
print("Resnet34 U-Net with Imagenet weights for multi-class data finished.")
execution_time = stop-start
print(
    "Execution time for Resnet34 U-Net with Imagenet weights for multi-class data:", 
      execution_time)
log_execution_time(
    f"unet_resnet_imagenet_multi_{epochs}ep_B{batch_size}:{loss_n}loss", 
    execution_time, # choose path for current case
    # stat_output_dir_test_fits, 
    stat_output_dir_final_fits
)
unet_resnet_imagenet_multi.save(
    os.path.join( # choose path for current case
        # model_output_dir_test_fits,
        model_output_dir_final_fits,
        f"unet_resnet_imagenet_multi_{epochs}ep_B{batch_size}_{loss_n}loss.hdf5"
    )
)
save_hist_as_csv(
    hist_unet_resnet_imagenet_multi, 
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"unet_resnet_imagenet_multi_{epochs}ep_B{batch_size}_{loss_n}loss_hist.csv"
    )
)


'''
Residual U-Net with attention
'''

# adapt epochs and batch size to case and memory
epochs = 50
# epochs = 100
batch_size = 16

loss_n = "catfocCE"
metr_n = "mIoU"

att_res_unet_multi = get_att_res_unet (input_shape, n_classes)
metric = tf.keras.metrics.MeanIoU(num_classes=n_classes)
att_res_unet_multi.compile(
    optimizer=Adam(learning_rate = 1e-2), 
    loss=keras.losses.CategoricalFocalCrossentropy(), 
    metrics=[metric])
print(att_res_unet_multi.summary())
print("Residual U-Net with attention for mulit-class data running.")
start = datetime.now() 
hist_att_res_unet_multi = att_res_unet_multi.fit(
    X_train, y_train, 
    verbose=1,
    batch_size = batch_size,
    validation_data=(X_val, y_val), 
    shuffle=True,
    epochs=epochs)
stop = datetime.now()
print("Residual U-Net with attention for multi-class data finished.")
execution_time = stop-start
print(
    "Execution time for Residual U-Net with attention for multi-class data:", 
    execution_time)
log_execution_time(
    f"att_res_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss", 
    execution_time, # choose path for current case
    # stat_output_dir_test_fits, 
    stat_output_dir_final_fits
)
att_res_unet_multi.save(
    os.path.join( # choose path for current case
        # model_output_dir_test_fits,
        model_output_dir_final_fits,
        f"att_res_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss.hdf5"
    )
)
save_hist_as_csv(
    hist_att_res_unet_multi, 
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"att_res_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss_hist.csv"
    )
)

csv_dir = os.path.join( # choose path for current case
    # stat_output_dir_test_fits,
    stat_output_dir_final_fits,
    f"att_res_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss_hist.csv")
data = pd.read_csv(csv_dir)

# Plotting metric plot
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['mean_io_u'], label='Mean IoU')
plt.plot(data.index, 
         data['val_mean_io_u'], 
         label='Validation Mean IoU', 
         linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Mean IoU')
# Choose title for current case
# plt.title('Multi-class residual U-net with attention: training and validation metric of test run')
plt.title('Multi-class residual U-net with attention: training and validation metric of final run')
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_test_fits,
        f"att_res_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss_{metr_n}_plot.png"),
    dpi=300)
plt.show()

# Plotting Loss
plt.figure(figsize=(10, 5))
plt.plot(data.index, 
         data['loss'], 
         label='Loss')
plt.plot(data.index, 
         data['val_loss'], 
         label='Validation Loss', 
         linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# Choose title for current case
# plt.title('Multi-class residual U-net with attention: training and validation loss of test run')
plt.title('Multi-class residual U-net with attention: training and validation loss of final run')
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join( # choose path for current case
        # stat_output_dir_test_fits,
        stat_output_dir_final_fits,
        f"att_res_unet_multi_{epochs}ep_B{batch_size}_{loss_n}loss_plot.png"), 
        dpi=300)
plt.show()