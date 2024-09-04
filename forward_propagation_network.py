import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs
    network = {}

    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]

        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes

    return network

def compute_weighted_sum(inputs, weights, bias):
    weighted_sum = np.sum(inputs * weights) + bias
    print('Weighted sum:', weighted_sum)
    return weighted_sum

def forward_propagate(network, inputs):
    layer_inputs = inputs.copy()

    for layer in network:
        layer_data = network[layer]
        layer_outputs = []

        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            node_weighted_sum = compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias'])
            node_output = sigmoid(node_weighted_sum)
            layer_outputs.append(np.around(node_output[0], decimals=4))
            print('Node {} output: {}'.format(layer_node, np.around(node_output[0], decimals=4)))

        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))

        layer_inputs = layer_outputs

    return layer_outputs

# Example usage:
num_inputs = 5
num_hidden_layers = 3
num_nodes_hidden = [3, 2, 3]
num_nodes_output = 1

my_network = initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output)

inputs = np.around(np.random.uniform(size=num_inputs), decimals=2)
print('The inputs to the network are {}'.format(inputs))

predictions = forward_propagate(my_network, inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))
