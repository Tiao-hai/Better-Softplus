# coding=utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Setting the random seed
seed = 50
np.random.seed(seed)
tf.random.set_seed(seed)

# Softplus
# Define the neural network structure
# def simple_neural_network(x, weights, biases):
#     layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
#     output_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
#     return output_layer

# Better-Softplus
def custom_activation(x):
    return 2*tf.math.log(1 + 0.2*x)

# Define the neural network structure
def simple_neural_network(x, weights, biases):
    layer_1 = custom_activation(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    output_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
    return output_layer


# Data generation
x_data = np.random.rand(1000, 1).astype(np.float32)
y_data = (np.sin(x_data * 2 * np.pi) + 0.1 * np.random.randn(*x_data.shape)).astype(np.float32)

# Initialising weights and biases
weights = {
    'h1': tf.Variable(tf.random.normal([1, 10], dtype=tf.float32), name='hidden_weights'), # Hidden layer weight
    'out': tf.Variable(tf.random.normal([10, 1], dtype=tf.float32), name='output_weights') # Output layer weight
}
biases = {
    'b1': tf.Variable(tf.zeros([10], dtype=tf.float32)), # Hidden Layer Bias
    'out': tf.Variable(tf.zeros([1], dtype=tf.float32))  # Output Layer Bias
}

# Setting up the Optimiser
optimizer = tf.optimizers.Adam()

# Training Models
epochs = 100
convergence_plot = []
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = simple_neural_network(x_data, weights, biases)
        loss_value = tf.reduce_mean(tf.square(predictions - y_data))
    gradients = tape.gradient(loss_value, [*weights.values(), *biases.values()])
    optimizer.apply_gradients(zip(gradients, [*weights.values(), *biases.values()]))
    convergence_plot.append(loss_value.numpy())

# Plotting loss curves
# Plot the change in loss values for each epoch to show the convergence of the model.
plt.plot(range(epochs), convergence_plot)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Activation Function Convergence (Custom Activation)')
plt.show()

# Output data
output_file_path = r'C:\Users\user\Desktop\1.txt'
with open(output_file_path, 'w') as f:
    for epoch in range(epochs):
        f.write(f'{epoch}  {convergence_plot[epoch]}\n')