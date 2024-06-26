from typing import List
import torch
import torch.nn as nn
from efficient_kan import KAN

def build_kan(
        param_list: List[int],
        ) -> nn.Module:
  """Build a MultiLayer Perceptron.

  Args:
    input_size: Size of input layer.
    layer_sizes: An array of input size for each hidden layer.
    output_size: Size of the output layer.
    output_activation: Activation function for the output layer.
    activation: Activation function for the hidden layers.

  Returns:
    mlp: An MLP sequential container.
  """
  model = KAN(param_list)

  return model
  # Size of each layer
#   layer_sizes = [input_size] + hidden_layer_sizes
#   if output_size:
#     layer_sizes.append(output_size)

#   # Number of layers
#   nlayers = len(layer_sizes) - 1

#   # Create a list of activation functions and
#   # set the last element to output activation function
#   act = [activation for i in range(nlayers)]
#   act[-1] = output_activation

#   # Create a torch sequential container
#   mlp = nn.Sequential()
#   for i in range(nlayers):
#     mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
#                                              layer_sizes[i + 1]))
#     mlp.add_module("Act-" + str(i), act[i]())
# return mlp