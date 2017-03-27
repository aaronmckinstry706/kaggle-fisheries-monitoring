# All variables and functions are named according to machine learning
# terminology: example, input, output, label, etc. For example, an input could be
# a scalar, or a vector, or a matrix, or a tensor.

import logging
import os

import lasagne
import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import lasagne.objectives as objectives
import lasagne.updates as updates
import numpy
import theano
import theano.tensor as tensor

import loader

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# Set the preprocessing parameters.
image_width = 512

# Set the learning algorithm parameters.
learning_rate = theano.shared(numpy.float32(0.01))
momentum = 0.9
batch_size = 64

# Set some architectural parameter invariants.
num_classes = 8

# Define the symbolic variables and expressions for network computation.
inputs = tensor.tensor4(name='inputs')
labels = tensor.matrix(name='labels')

input_layer = layers.InputLayer(shape=(None, 3, image_width, image_width),
                                input_var=inputs, name='input_layer')
input_layer_reshaped = layers.ReshapeLayer(incoming=input_layer,
                                           shape=(image_width*image_width*3,),
                                           name='input_layer_reshaped')
output_layer = layers.DenseLayer(
    incoming=input_layer_reshaped, num_units=8,
    nonlinearity=nonlinearities.softmax, name='output_layer')

network_parameters = layers.get_all_params(output_layer)
outputs = layers.get_output(output_layer)
unaveraged_losses = tensor.sum(tensor.sum(objectives.squared_error(outputs,
                                                                   labels)))
loss = tensor.sum(unaveraged_losses) / (batch_size*8)

logger.info('Compiling the train, test, and validation functions...')

network_updates = updates.nesterov_momentum(
    loss_or_grads=loss,
    params=network_parameters,
    learning_rate=learning_rate,
    momentum=momentum)
train = theano.function(
    inputs=[inputs, labels],
    outputs=[outputs, loss],
    updates=network_updates)
validate = theano.function(inputs=[inputs, labels], outputs=[outputs, loss])
test = theano.function(inputs=[inputs], outputs=[outputs])



logger.info('Finished.')