# import numpy as np
# import matplotlib.pyplot as plt
# import os

# save_path = "/data_hdd/pauline/plots/activation_functions/"

# # Define activation functions
# def relu(x):
#     return np.maximum(0, x)

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)

# # Generate x values
# x = np.linspace(-10, 10, 500)

# # Compute activation values
# relu_values = relu(x)
# sigmoid_values = sigmoid(x)
# softmax_values = softmax(np.array([x, -x]))

# # Define line style for x=0
# zero_line_style = dict(color="darkgrey", linewidth=1.5, linestyle="--")

# # Define font sizes for the plot
# title_fontsize = 18
# axis_title_fontsize = 16
# axis_label_fontsize = 14
# legend_fontsize = 14

# # Create subplots side-by-side with updated font sizes
# fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# # Plot ReLU
# ax[0].plot(x, relu_values, label="ReLU", color="blue")
# ax[0].axvline(0, **zero_line_style)
# ax[0].set_title("a) ReLU Function: f(x) = max(0, x)", fontsize=title_fontsize)
# ax[0].set_xlabel("x", fontsize=axis_title_fontsize)
# ax[0].set_ylabel("f(x)", fontsize=axis_title_fontsize)
# ax[0].tick_params(axis='both', labelsize=axis_label_fontsize)
# ax[0].grid(True)
# ax[0].legend(fontsize=legend_fontsize)

# # Plot Sigmoid
# ax[1].plot(x, sigmoid_values, label="Sigmoid", color="green")
# ax[1].axvline(0, **zero_line_style)
# ax[1].set_title("b) Sigmoid Function: f(x) = 1 / (1 + exp(-x))", fontsize=title_fontsize)
# ax[1].set_xlabel("x", fontsize=axis_title_fontsize)
# ax[1].set_ylabel("f(x)", fontsize=axis_title_fontsize)
# ax[1].tick_params(axis='both', labelsize=axis_label_fontsize)
# ax[1].grid(True)
# ax[1].legend(fontsize=legend_fontsize)

# # Plot Softmax (only positive component)
# ax[2].plot(x, softmax_values[0], label="Softmax", color="red")
# ax[2].axvline(0, **zero_line_style)
# ax[2].set_title("c) Softmax Function: f(x_i) = exp(x_i) / Σ(exp(x_j))", fontsize=title_fontsize)
# ax[2].set_xlabel("x", fontsize=axis_title_fontsize)
# ax[2].set_ylabel("f(x)", fontsize=axis_title_fontsize)
# ax[2].tick_params(axis='both', labelsize=axis_label_fontsize)
# ax[2].grid(True)
# ax[2].legend(fontsize=legend_fontsize)

# # Adjust layout and display
# plt.tight_layout()

# # Save the plot as a file
# plt.savefig(os.path.join(save_path,
#                          "activation_functions_comparison.png"), dpi=300)

# # Show the plot
# plt.show()

###############################################################################
# plot model architectures
import os
import numpy as np
import matplotlib.pyplot as plt
from simple_unet_model import simple_unet_model
from simple_multi_unet_model import multi_unet_model
from res_unet_attention import Attention_ResUNet
from tensorflow.keras.utils import plot_model

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 4

# simp unet binary
def get_simple_unet_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
simp_unet_bin = get_simple_unet_model()
plot_model(
    simp_unet_bin, 
    to_file="/data_hdd/pauline/swf/results/model_archs/sim_unet_bin_arch.png")

# Residual U-Net with attention binary
input_shape = (256, 256, 4)
n_classes = 2
loss_n = "binfoc"
metr_n = "binIoU"

def get_att_res_unet (input_shape, n_classes):
    return Attention_ResUNet(input_shape, n_classes)

att_res_unet_bin = get_att_res_unet (input_shape, n_classes)

plot_model(
    att_res_unet_bin, 
    to_file="/data_hdd/pauline/swf/results/model_archs/att_res_unet_bin_arch.png")


# Simple U-Net multi class
def get_multi_unet_model(n_classes):
    return multi_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

n_classes = 4

simp_unet_multi = get_multi_unet_model(n_classes)

plot_model(
    simp_unet_multi,
    to_file="/data_hdd/pauline/swf/results/model_archs/simp_unet_multi_arch.png")


# Residual U-Net with attention multi

att_res_unet_multi = get_att_res_unet (input_shape, n_classes)

plot_model(
    att_res_unet_multi,
    to_file="/data_hdd/pauline/swf/results/model_archs/att_res_unet_multi_arch.png")

