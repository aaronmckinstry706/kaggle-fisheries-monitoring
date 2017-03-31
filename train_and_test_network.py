# All variables and functions are named according to machine learning
# terminology: example, input, output, label, etc. For example, an input could be
# a scalar, or a vector, or a matrix, or a tensor.

import logging
import os
import time

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
image_width = 256

# Set the learning algorithm parameters.
learning_rate = theano.shared(numpy.float32(0.01))
momentum = 0.9
batch_size = 64

# Set architectural parameter invariants.
num_classes = 8

# Define the symbolic variables and expressions for network computations.
inputs = tensor.tensor4(name='inputs')
labels = tensor.matrix(name='labels')

input_layer = layers.InputLayer(shape=(None, 3, image_width, image_width),
                                input_var=inputs)

hidden_layer_one_no_batchnorm = layers.Conv2DLayer(incoming=input_layer,
                                                   num_filters=64,
                                                   filter_size=(3,3),
                                                   pad='same')
hidden_layer_one = layers.batch_norm(hidden_layer_one_no_batchnorm)

output_layer = layers.DenseLayer(
    incoming=hidden_layer_one, num_units=num_classes,
    nonlinearity=nonlinearities.softmax)

train_outputs = layers.get_output(output_layer, deterministic=False)
test_outputs = layers.get_output(output_layer, deterministic=True)

network_parameters = layers.get_all_params(output_layer, trainable=True)

train_loss = tensor.sum(
    objectives.categorical_crossentropy(train_outputs, labels)) / batch_size
validation_loss = objectives.categorical_crossentropy(test_outputs, labels)

network_updates = updates.nesterov_momentum(
    loss_or_grads=train_loss,
    params=network_parameters,
    learning_rate=learning_rate,
    momentum=momentum)

logger.info('Compiling the train and validation functions...')

train = theano.function(
    inputs=[inputs, labels],
    outputs=[train_outputs, train_loss],
    updates=network_updates)
validate = theano.function(inputs=[inputs, labels],
                           outputs=[test_outputs, validation_loss])

logger.info('Initializing data loader...')

data_loader = loader.Loader(batch_size, image_width,
                            'data/train',
                            'data/test')

logger.info('Pre-loading data...')

data_loader.get_shuffled_train_input_iterator()
data_loader.get_validation_input_iterator()
data_loader.get_test_input_iterator()

logger.info('Training model...')

num_epochs = 100

for i in range(0, num_epochs):
    epoch_start_time = time.time()
    
    train_iterator = data_loader.get_shuffled_train_input_iterator()
    avg_batch_train_loss = 0
    num_iterations = 0
    for images_labels in train_iterator:
        outputs, training_loss = train(images_labels[0], images_labels[1])
        avg_batch_train_loss += training_loss
        num_iterations += 1
    avg_batch_train_loss /= num_iterations
    
    validation_iterator = data_loader.get_validation_input_iterator()
    avg_validate_loss = 0
    num_examples = 0
    for images_labels in validation_iterator:
        outputs, validation_loss = validate(images_labels[0], images_labels[1])
        avg_validate_loss += numpy.sum(validation_loss)
        num_examples += images_labels[0].shape[0]
    avg_validate_loss /= num_examples
    
    epoch_end_time = time.time()
    
    print('Epoch ' + str(i) + ': ')
    print('    Average batch batch_crossentropy_loss over validation set: '
          + str(avg_validate_loss) + '.')
    print('    Average batch batch_crossentropy_loss over training set: '
          + str(avg_batch_train_loss) + '.')
    print('    Time: ' + str(epoch_end_time - epoch_start_time) + ' seconds.')

logger.info('Finished.')