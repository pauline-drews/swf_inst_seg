import numpy as np
import matplotlib.pyplot as plt
import os

save_path = "/data_hdd/pauline/plots/activation_functions/"

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Generate x values
x = np.linspace(-10, 10, 500)

# Compute activation values
relu_values = relu(x)
sigmoid_values = sigmoid(x)

# Simplify softmax for visualization
softmax_values = softmax(np.array([x, -x]))

# Define line style for x=0
zero_line_style = dict(color="darkgrey", linewidth=1.5, linestyle="--")

# Create subplots side-by-side
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Plot ReLU
ax[0].plot(x, relu_values, label="ReLU", color="blue")
ax[0].axvline(0, **zero_line_style)
ax[0].set_title("ReLU Function: f(x) = max(0, x)")
ax[0].set_xlabel("x")
ax[0].set_ylabel("f(x)")
ax[0].grid(True)
ax[0].legend()

# Plot Sigmoid
ax[1].plot(x, sigmoid_values, label="Sigmoid", color="green")
ax[1].axvline(0, **zero_line_style)
ax[1].set_title("Sigmoid Function: f(x) = 1 / (1 + exp(-x))")
ax[1].set_xlabel("x")
ax[1].set_ylabel("f(x)")
ax[1].grid(True)
ax[1].legend()

# Plot Softmax (only positive component)
ax[2].plot(x, softmax_values[0], label="Softmax", color="red")
ax[2].axvline(0, **zero_line_style)
ax[2].set_title("Softmax Function: f(x) = exp(x) / sum(exp(x))")
ax[2].set_xlabel("x")
ax[2].set_ylabel("f(x)")
ax[2].grid(True)
ax[2].legend()

# Adjust layout and display
plt.tight_layout()

# Save the plot as a file
plt.savefig(os.path.join(save_path,
                         "activation_functions_comparison.png"), dpi=300)

# Show the plot
plt.show()