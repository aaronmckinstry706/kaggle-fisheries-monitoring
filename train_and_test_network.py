# All variables and functions are named according to machine learning
# terminology: example, input, output, label, etc. For example, an input could be
# a scalar, or a vector, or a matrix, or a tensor.

import datetime
import logging
import os
import sys
import time

import keras.preprocessing.image
import lasagne
import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import lasagne.objectives as objectives
import lasagne.updates as updates
import numpy
import theano
import theano.tensor as tensor

import utilities
import preprocessing

def get_param_description_str(**kwargs):
    return str(kwargs)

def get_learning_rate(
        training_loss_history,
        validation_loss_history,
        learning_rate):
    if len(training_loss_history) % 100 == 0:
        return numpy.float32(learning_rate/10.0)
    else:
        return numpy.float32(learning_rate)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
#logger.addHandler(logging.FileHandler('logs/network_training.log'))
logger.setLevel(logging.DEBUG)
logger.info('\nStarting run at ' + str(datetime.datetime.now()) + '.')

training_directory = 'data/train'
validation_directory='data/validation'
test_directory='data/test_stg1'

num_threads_for_preprocessing = 16

# Set the preprocessing parameters.
image_width = 512

# Set the learning algorithm parameters.
learning_rate = theano.shared(numpy.float32(0.001))
momentum = 0.9
batch_size = 64
weight_decay = 0.0001

# Set architectural parameter invariants.
num_classes = 8

# Define the symbolic variables and expressions for network computations.
inputs = tensor.tensor4(name='inputs')
labels = tensor.matrix(name='labels')

input_layer = layers.InputLayer(shape=(None, 3, image_width, image_width),
                                input_var=inputs)

hidden_layer_one = layers.Conv2DLayer(incoming=input_layer,
                                      num_filters=16,
                                      filter_size=(3,3),
                                      stride=(2,2),
                                      pad='same')
hidden_layer_one = layers.batch_norm(hidden_layer_one)

hidden_layer_two = layers.Conv2DLayer(incoming=hidden_layer_one,
                                      num_filters=32,
                                      filter_size=(3,3),
                                      stride=(2,2),
                                      pad='same')
hidden_layer_two = layers.batch_norm(hidden_layer_two)

hidden_layer_three = layers.Conv2DLayer(incoming=hidden_layer_two,
                                        num_filters=64,
                                        filter_size=(3,3),
                                        pad='same')
hidden_layer_three = layers.batch_norm(hidden_layer_three)

hidden_layer_three = layers.Conv2DLayer(incoming=hidden_layer_two,
                                        num_filters=8,
                                        filter_size=(3,3),
                                        pad='same')
hidden_layer_three = layers.batch_norm(hidden_layer_three)

output_map_pooling_layer = layers.GlobalPoolLayer(incoming=hidden_layer_three,
                                                  pool_function=tensor.max)

output_layer = layers.NonlinearityLayer(incoming=output_map_pooling_layer,
                                        nonlinearity=nonlinearities.softmax)

train_outputs = layers.get_output(output_layer, deterministic=False)
test_outputs = layers.get_output(output_layer, deterministic=True)

network_parameters = layers.get_all_params(output_layer, trainable=True)

train_loss_without_regularization = tensor.sum(
    objectives.categorical_crossentropy(train_outputs, labels)) / batch_size
train_loss = train_loss_without_regularization
for trainable_layer_parameters in network_parameters:
    current_decay_term = tensor.sum(
        trainable_layer_parameters*trainable_layer_parameters)
    train_loss = train_loss + weight_decay*current_decay_term
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

logger.info('Splitting training and validation images...')

utilities.separate_validation_set(training_directory, validation_directory,
                                  split=0.1)

logger.info('Training model...')

num_epochs = 300
previous_validate_losses = []
previous_train_losses = []
previous_learning_rates = [(0,learning_rate.get_value())]
previous_learning_rate = None

logger.info(get_param_description_str(
    image_width=image_width,
    learning_rate=learning_rate.get_value(),
    batch_size=batch_size,
    weight_decay=weight_decay))

for i in range(0, num_epochs):
    epoch_start_time = time.time()
    
    train_iterator = preprocessing.get_threaded_generator(
        training_directory, image_width, batch_size, type='training',
        num_threads=num_threads_for_preprocessing)
    avg_batch_train_loss = 0
    num_iterations = 0
    for images_labels in train_iterator:
        outputs, training_loss = train(numpy.moveaxis(images_labels[0],
                                                      3, 1),
                                       images_labels[1])
        avg_batch_train_loss += training_loss
        num_iterations += 1
        logger.debug('num_iterations = ' + str(num_iterations))
    avg_batch_train_loss /= num_iterations
    
    validation_iterator = preprocessing.get_threaded_generator(
        validation_directory, image_width, batch_size, type='validation',
        num_threads=num_threads_for_preprocessing)
    avg_validate_loss = 0
    num_examples = 0
    for images_labels in validation_iterator:
        outputs, validation_loss = validate(numpy.moveaxis(images_labels[0],
                                                           3, 1),
                                            images_labels[1])
        avg_validate_loss += numpy.sum(validation_loss)
        num_examples += images_labels[0].shape[0]
        logger.debug('num_examples = ' + str(num_examples))
    avg_validate_loss /= num_examples
    
    epoch_end_time = time.time()
    
    previous_validate_losses.append(avg_validate_loss)
    previous_train_losses.append(avg_batch_train_loss)
    previous_learning_rate = learning_rate.get_value()
    learning_rate.set_value(get_learning_rate(previous_train_losses,
                                              previous_validate_losses,
                                              learning_rate.get_value()))
    logger.info('Epoch ' + str(i) + ':\n'
                + '    Crossentropy loss over validation set: '
                + str(avg_validate_loss) + '.\n'
                + '    Average batch crossentropy loss over training set:'
                + ' ' + str(avg_batch_train_loss) + '.\n'
                + '    Time: '
                + str(int(round(epoch_end_time - epoch_start_time)))
                + ' seconds.')
    
    if previous_learning_rate != learning_rate.get_value():
        logger.info('Changing learning rate from '
                    + str(previous_learning_rate) + ' to '
                    + str(learning_rate.get_value()) + '.')
        previous_learning_rates.append((i+1, learning_rate.get_value()))

logger.info('Merging validation folder back into training folder...')

utilities.recombine_validation_and_training(validation_directory,
                                            training_directory)

logger.info('Average batch training loss per epoch: '
            + str(previous_validate_losses))
logger.info('Validation loss per epoch: '
            + str(previous_train_losses))
logger.info('Learning rate schedule: ' + str(previous_learning_rates))
logger.info('Finishing run at ' + str(datetime.datetime.now()) + '.')
