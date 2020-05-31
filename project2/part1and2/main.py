import sys
sys.path.append("..")
import utils
from utils import *
from softmax_skeleton import softmax_regression, get_classification, plot_cost_function_over_time, compute_test_error, compute_test_error_mod3, update_y
import numpy as np
import matplotlib.pyplot as plt
from features import *

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
plot_images(train_x[0:20,:])


# TODO: first fill out functions in softmax_skeleton.py, or run_softmax_on_MNIST will not work

# run_softmax_on_MNIST: trains softmax, classifies test data, computes test error, and plots cost function
def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")
    
    # TODO: add your code here for the "Using the Current Model" question in tab 4.
    #      and print the testErrorMod3
    return test_error

# Don't run this until the relevant functions in softmax_skeleton.py have been fully implemented.
#print('testError =', run_softmax_on_MNIST(temp_parameter=1))

# TODO: Find the error rate for temp_parameter = [.5, 1.0, 2.0]
#      Remember to return the tempParameter to 1, and re-run run_softmax_on_MNIST


def run_softmax_on_MNIST_mod3(temp_parameter):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)
    train_y,test_y = update_y(train_y,test_y)
    test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    return test_error
#for temp in [.5,1.0,2.0]:
    #print(run_softmax_on_MNIST_mod3(temp))


# TODO: Run run_softmax_on_MNIST_mod3(), report the error rate



######################################################
# This section contains the primary code to run when
# working on the "Using manually crafted features" part of the project.
# You should only work on this section once you have completed the first part of the project.
######################################################
## Dimensionality reduction via PCA ##

# TODO: First fill out the PCA functions in features.py as the below code depends on them.
train_x, train_y, test_x, test_y = get_MNIST_data()
n_components = 18
pcs = principal_components(train_x)
train_pca = project_onto_PC(train_x, pcs, n_components)
test_pca = project_onto_PC(test_x, pcs, n_components)
# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.
theta, cost_function_history = softmax_regression(train_pca, train_y, temp_parameter=1, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)
test_error = compute_test_error(test_pca, test_y, theta, temp_parameter=1)
print(test_error)

# TODO: Use the plot_PC function in features.py to produce scatterplot
#       of the first 100 MNIST images, as represented in the space spanned by the
#       first 2 principal components found above.
plot_PC(train_x[range(100),], pcs, train_y[range(100)])


# TODO: Use the reconstruct_PC function in features.py to show
#       the first and second MNIST images as reconstructed solely from
#       their 18-dimensional principal component representation.
#       Compare the reconstructed images with the originals.
firstimage_reconstructed = reconstruct_PC(train_pca[0,], pcs, n_components, train_x)
plot_images(firstimage_reconstructed)
plot_images(train_x[0,])

secondimage_reconstructed = reconstruct_PC(train_pca[1,], pcs, n_components, train_x)
plot_images(secondimage_reconstructed)
plot_images(train_x[1,])



## Cubic Kernel ##
# TODO: Find the 10-dimensional PCA representation of the training and test set


# TODO: First fill out cubicFeatures() function in features.py as the below code requires it.

train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)
# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.


# TODO: Train your softmax regression model using (train_cube, train_y)
#       and evaluate its accuracy on (test_cube, test_y).
