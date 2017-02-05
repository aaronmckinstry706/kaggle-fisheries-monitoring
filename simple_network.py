# All variables and functions are named according to machine learning terminology: example, input,
# output, label, etc. For example, an input could be a scalar, or a vector, or a matrix, or a
# tensor. 

import theano
import theano.tensor as tensor
import numpy
import lasagne
import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import lasagne.updates as updates
import lasagne.objectives as objectives

# Set the preprocessing and learning parameters. 
batch_size = 64
image_width = 64
learning_rate = 1.0
momentum = 0.9

# Define the symbolic variables and expressions for network computation. 
inputs = tensor.tensor3(name='inputs')
labels = tensor.matrix(name='labels')

input_layer = layers.InputLayer(shape=(None, image_width, image_width), input_var=inputs,
  name='input_layer')
input_layer_reshaped = layers.ReshapeLayer(incoming=input_layer, shape=(64*64,),
  name='input_layer_reshaped')
output_layer = layers.DenseLayer(incoming=input_layer_reshaped, num_units=8,
  nonlinearity=nonlinearities.softmax, name='output_layer')

network_parameters = layers.get_all_params(output_layer)
outputs = layers.get_output(output_layer)
loss = tensor.sum(tensor.sum(tensor.sum(objectives.squared_error(outputs, labels)))) \
  / (batch_size*image_width*image_width)

# Compile the train, test, and validation functions
network_updates = updates.nesterov_momentum(loss_or_grads=loss, params=network_parameters,
  learning_rate=learning_rate, momentum=momentum)
train = theano.function(inputs=[inputs, labels], outputs=[outputs, loss],
  updates=network_updates)
validate = theano.function(inputs=[inputs, labels], outputs=[outputs, loss])
test = theano.function(inputs=[inputs], outputs=[outputs])

