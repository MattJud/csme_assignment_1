#!/usr/bin/env python
# coding: utf-8

# # Exercise 1:
# In this exercise you will program a simple neural network with numpy. The tasks will be:
# 1. Implementing a Forward propagation
# 2. Completing the Forward propagation by computing the loss
# 3. Implementing a Back propagation
# 4. Training the neural network
#
# You will implement code in 'nn.py' and test your implementations with the code provided below.


# This code initializes the network (see __init__ in nn.py) and some other functions needed later. 
# The input data (X) with the associated labels (y) as well as the weights and biases 
# are initialized with random numbers. A seed is set to make your results comparable.

import numpy as np
import matplotlib.pyplot as plt
from nn import TwoLayerNet

input_dim = 4
hidden_dim = 10
num_classes = 3
num_inputs = 5

def init_net():
    np.random.seed(0)
    return TwoLayerNet(input_dim, hidden_dim, num_classes, std=1e-1)

def init_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_dim)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# @Matthew: probably must be used in the Notebook 
#get_ipython().run_line_magic('matplotlib', 'inline') 
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net = init_net()
X, y = init_data()


# # Forward propagation
# Go to 'nn.py' to implement the Forward propagation in the loss_grad function (Task 1.1 and Task 1.2).
# Then test the code below to check your results.


scores = net.loss_grad(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# The difference should be very small. (< 1e-7)
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))


# # Time difference
# Add code to measure the time of your naive implementation with at least 2 loops and the implementation with no loops and fill in your results below. 
# Note that 'time' is already imported in the 'nn.py' file which you can use for the time measurement. 

# Duration with 2 loops:
0.008005380630493164

# Duration without loops:
0.0

# # Forwardpass loss
# Complete the forward propagation by implementing the loss function in 'nn.py' (Task 2).
# Run the code below to check your implementation.


loss, _ = net.loss_grad(X, y, reg=0.05)
correct_loss = 1.30378789133

# should be very small, (< 1e-12)
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))


# # Numerical issues with Softmax
# 
# Please describe the two main issues that can lead to numerical instability when using the softmax function? Describe a/your approach of avoiding instabilities with softmax.

# TODO: @All: Issues are exploding or vanishing class scores. solution --> add a constant to the function (see link below)
#  https://eulertech.wordpress.com/2017/10/09/numerical-instability-in-deep-learning-with-softmax/

# # Back propagation
# Now implement the Back propagation in 'nn.py' (Task 3) in the loss_grad function.
# Test the code below to check your results.


from eval_grad import numerical_grad

# Here we use a numeric gradient to check your implementation of the back propagation.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss_grad(X, y, reg=0.05)

# these should all be less than 1e-8
for param_name in grads:
    f = lambda W: net.loss_grad(X, y, reg=0.05)[0]
    param_grad_num = numerical_grad(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


# # Training a network
# Finaly we want to train our network. For this purpose go to 'nn.py' and implement the train and predict function (Task 4.1, 4.2 and 4.3).
# Then test the code below to check your results.

'''
net = init_net()
stats = net.train(X, y, X, y,
                  learning_rate=1e-1, reg=5e-6,
                  num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()


# # Iterations
# Please explain at least how many iterations are useful for training this neural network (use the plot above for your explanation)?
'''





