import numpy as np

def last_layer_error(target, output, weighted_inputs, activation_der, cost_der):
    error = cost_der(target, output) * activation_der(weighted_inputs)
    # print("Last Error: ", error)
    return error

def layer_error(next_layer_weights, next_layer_error, weighted_inputs, activation_der):
    # print("LayerError Multiplying:\n", next_layer_weights.T, "\nby\n", next_layer_error)
    error = (next_layer_weights.T @ next_layer_error) * activation_der(weighted_inputs)
    # print("Error: ", error)
    return error

