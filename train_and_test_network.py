import datetime
import logging
import os
import sys
import time
import typing

import keras.preprocessing.image
import lasagne
import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import lasagne.objectives as objectives
import lasagne.updates as updates
import numpy
import theano
import theano.tensor as tensor

import config
import network_architectures as architectures
import preprocessing
import utilities

def get_learning_rate(training_loss_history,    # type: typing.List[float]
                      validation_loss_history,  # type: typing.List[float]
                      learning_rate             # type: float
                      ):
    # type: (...) -> float
    if len(training_loss_history) % 200 == 0 and len(training_loss_history) > 0:
        return learning_rate/10.0
    else:
        return learning_rate

logger = logging.getLogger(__name__) # type: logging.Logger
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('logs/network_training.log'))
logger.setLevel(logging.INFO)
logger.info('\nStarting run at ' + str(datetime.datetime.now()) + '.')

params = config.read_config_file('config.txt')

# Define the symbolic variables and expressions for network computations.
learning_rate = theano.shared(numpy.float32(params['initial_learning_rate']))
inputs = tensor.tensor4(name='inputs')
labels = tensor.matrix(name='labels')

output_layer = architectures.fully_convolutional_network(
    inputs=inputs,
    image_shape=(3, params['image_width'], params['image_width']),
    num_outputs=params['num_classes'])

train_outputs = layers.get_output(output_layer, deterministic=False)
test_outputs = layers.get_output(output_layer, deterministic=True)

network_parameters = layers.get_all_params(output_layer, trainable=True)

train_loss_without_regularization = tensor.sum(
    objectives.categorical_crossentropy(train_outputs, labels)) / params['batch_size']
train_loss = train_loss_without_regularization + lasagne.regularization.l2(
    train_outputs)
validation_loss = objectives.categorical_crossentropy(test_outputs, labels)

network_updates = updates.nesterov_momentum(
    loss_or_grads=train_loss,
    params=network_parameters,
    learning_rate=learning_rate,
    momentum=params['momentum'])

logger.info('Compiling the train and validation functions.')
train = theano.function(
    inputs=[inputs, labels],
    outputs=[train_outputs, train_loss],
    updates=network_updates)
validate = theano.function(inputs=[inputs, labels],
                           outputs=[test_outputs, validation_loss])

logger.info('Cleaning up training/validation split from previous runs.')
utilities.recombine_validation_and_training(params['validation_directory'],
                                            params['training_directory'])

logger.info('Splitting training and validation images.')
utilities.separate_validation_set(params['training_directory'], params['validation_directory'],
                                  split=0.1)

logger.info('Training model.')

previous_validate_losses = []   # type: typing.List[float]
previous_train_losses = []      # type: typing.List[float]
previous_learning_rates = []    # type: typing.List[typing.Tuple[int, float]]
previous_learning_rates.append((0,learning_rate.get_value()))
previous_learning_rate = None   # type: typing.Union[float, None]

logger.info(params)

for i in range(0, params['num_epochs']):
    epoch_start_time = time.time()
    
    train_iterator = preprocessing.get_generator(
        params['training_directory'], params['image_width'], params['batch_size'], type='training')
    threaded_train_iterator = preprocessing.get_threaded_generator(
        train_iterator, len(train_iterator.filenames),
        num_threads=params['num_threads_for_preprocessing'])
    avg_batch_train_loss = 0
    num_iterations = 0
    for images_labels in threaded_train_iterator:
        outputs, training_loss = train(numpy.moveaxis(images_labels[0],
                                                      3, 1),
                                       images_labels[1])
        avg_batch_train_loss += training_loss
        num_iterations += 1
    avg_batch_train_loss = avg_batch_train_loss / num_iterations
    
    validation_iterator = preprocessing.get_generator(
        params['validation_directory'], params['image_width'], params['batch_size'], type='validation')
    threaded_validation_iterator = preprocessing.get_threaded_generator(
        validation_iterator, len(validation_iterator.filenames),
        num_threads=params['num_threads_for_preprocessing'])
    avg_validate_loss = 0
    num_examples = 0
    for images_labels in threaded_validation_iterator:
        outputs, validation_loss = validate(numpy.moveaxis(images_labels[0],
                                                           3, 1),
                                            images_labels[1])
        avg_validate_loss += numpy.sum(validation_loss)
        num_examples += images_labels[0].shape[0]
    avg_validate_loss = avg_validate_loss / num_examples
    
    epoch_end_time = time.time()
    
    previous_validate_losses.append(avg_validate_loss)
    previous_train_losses.append(avg_batch_train_loss)
    previous_learning_rate = learning_rate.get_value()
    learning_rate.set_value(
        numpy.float32(
            get_learning_rate(
                previous_train_losses,
                previous_validate_losses,
                learning_rate.get_value())))
    
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
utilities.recombine_validation_and_training(params['validation_directory'],
                                            params['training_directory'])

logger.info('Average batch training loss per epoch: '
            + str(previous_validate_losses))
logger.info('Validation loss per epoch: '
            + str(previous_train_losses))
logger.info('Learning rate schedule: ' + str(previous_learning_rates))

logger.info('Finishing run at ' + str(datetime.datetime.now()) + '.')
