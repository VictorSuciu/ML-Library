import numpy as np

def last_layer_error(target, output, weighted_inputs, activation_der, cost_der):
    # print('before:', target, output, weighted_inputs)
    error = cost_der(target, output) * activation_der(weighted_inputs)
    # print('after:', target, output, weighted_inputs)
    # print('LAST curr_error = cost_der(target, output) * activation_der(weighted_inputs)\n', \
    # error, '\n=\n', cost_der(target, output), '\n*\n', activation_der(weighted_inputs), ('\n'+'-'*30+'\n'))
    
    return error

def layer_error(next_layer_weights, next_layer_error, weighted_inputs, activation_der):
    # print('before:', next_layer_weights, next_layer_error, weighted_inputs)
    error = (next_layer_weights.T @ next_layer_error) * activation_der(weighted_inputs)
    # print('after:', next_layer_weights, next_layer_error, weighted_inputs)
    # print('curr_error = (next_layer_weights.T @ next_layer_error) * activation_der(weighted_inputs)\n', \
    # error, '\n=\n(', next_layer_weights.T, '\n@\n', next_layer_error, '\n) *\n', activation_der(weighted_inputs),('\n'+'-'*30+'\n'))
    
    return error

