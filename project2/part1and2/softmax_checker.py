import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
from softmax_skeleton import augment_feature_vector, compute_probabilities, compute_cost_function, run_gradient_descent_iteration

def verify_input_output_types_softmax():
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x = augment_feature_vector(train_x[0:10])
    train_y = train_y[0:10]
    alpha = 0.3
    lambda_factor = 1.0e-4
    theta = np.zeros([10, train_x.shape[1]])
    temp_parameter = 1

    # check computeCostFunction
    cost = compute_cost_function(train_x, train_y, theta, lambda_factor, temp_parameter)
    print('PASSED: compute_cost_function appears to handle correct input type.')
    if isinstance(cost, float):
        print('PASSED: compute_cost_function appears to return the right type.')
    else:
        print('FAILED: compute_cost_function appears to return the wrong type. Expected {0} but got {1}'.format(float, type(cost)))

    # check computeProbabilities
    probabilities = compute_probabilities(train_x, theta, temp_parameter)
    print('PASSED: compute_probabilities appears to handle correct input type.')
    if isinstance(probabilities, np.ndarray):
        if probabilities.shape == (10, 10):
            print('PASSED: compute_probabilities return value appears to return a numpy array of the right shape.')
        else:
            print('FAILED: compute_probabilities return value appears to return a numpy array but with the wrong size. ' + \
                    'Expected a shape of {0} but got {1}.'.format((10,10), probabilities.shape))
    else:
        print('FAILED: compute_probabilities appears to be the wrong type. ' + \
                'Expected {0} but got {1}.'.format(type(np.array(range(4))), type(probabilities)))

    # check gradient descent
    theta = run_gradient_descent_iteration(train_x, train_y, theta, alpha, lambda_factor, temp_parameter)
    print('PASSED: run_gradient_descent_iteration appears to handle correct input type.')
    if isinstance(theta, np.ndarray):
        if theta.shape == (10, 785):
            print('PASSED: run_gradient_descent_iteration return value appears to return a numpy array of the right shape.')
        else:
            print('FAILED: run_gradient_descent_iteration return value appears to return a numpy array but with the wrong size. ' + \
                    'Expected {0} but got {1}.'.format((10, 785), theta.shape))
    else:
        print('FAILED: run_gradient_descent_iteration appears to return the wrong type. ' + \
                'Expected {0} but got {1}'.format(type(np.array(range(4))), type(theta)))


verify_input_output_types_softmax()
