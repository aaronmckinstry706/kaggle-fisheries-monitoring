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
import lasagne.regularization as regularization
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
    if len(training_loss_history) == 200:
        return learning_rate/10.0
    else:
        return learning_rate

logger = logging.getLogger(__name__) # type: logging.Logger
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('logs/network_training.log'))
logger.setLevel(logging.INFO)
logger.info('\nStarting run at ' + str(datetime.datetime.now()) + '.')

config_params = config.read_config_file('config.txt')

# Define the symbolic variables and expressions for network computations.
learning_rate = theano.shared(numpy.float32(config_params['initial_learning_rate']))
inputs = tensor.tensor4(name='inputs')
labels = tensor.matrix(name='labels')

network = architectures.fully_convolutional_network(
    inputs=inputs,
    image_shape=(3, config_params['image_width'], config_params['image_width']),
    num_outputs=config_params['num_classes'])

train_outputs = layers.get_output(network, deterministic=False)
test_outputs = layers.get_output(network, deterministic=True)

network_parameters = layers.get_all_params(network, trainable=True)

train_loss_without_regularization = tensor.sum(
    objectives.categorical_crossentropy(
        train_outputs, labels)) / config_params['batch_size']
train_loss = train_loss_without_regularization \
    + lasagne.regularization.regularize_network_params(
        layer=network, penalty=regularization.l2)
validation_loss = objectives.categorical_crossentropy(test_outputs, labels)

gradients = tensor.concatenate(
    [tensor.flatten(tensor.grad(train_loss, params))
     for params in layers.get_all_params(network,
                                         trainable=True)])
gradient_norm = gradients.norm(2)
#hessian = tensor.hessian(train_loss, params_as_vector)

network_updates = updates.nesterov_momentum(
    loss_or_grads=train_loss,
    params=network_parameters,
    learning_rate=learning_rate,
    momentum=config_params['momentum'])

# This is used in early stopping. The parameters that achieve the lowest
# error on the validation set are stored in this network.
best_network = architectures.fully_convolutional_network(
    inputs=inputs,
    image_shape=(3, config_params['image_width'], config_params['image_width']),
    num_outputs=config_params['num_classes'])

logger.info('Compiling the train and validation functions.')
train = theano.function(
    inputs=[inputs, labels],
    outputs=[train_outputs, train_loss],
    updates=network_updates)
validate = theano.function(inputs=[inputs, labels],
                           outputs=[test_outputs, validation_loss])
get_gradient_norm = theano.function(
    inputs=[inputs, labels],
    outputs=[gradient_norm])

logger.info('Cleaning up training/validation split from previous runs.')
utilities.recombine_validation_and_training(config_params['validation_directory'],
                                            config_params['training_directory'])

logger.info('Splitting training and validation images.')
utilities.separate_validation_set(config_params['training_directory'], config_params['validation_directory'],
                                  split=0.1)

logger.info('Training model.')

previous_validate_losses = []   # type: typing.List[float]
previous_train_losses = []      # type: typing.List[float]
previous_learning_rates = []    # type: typing.List[typing.Tuple[int, float]]
previous_learning_rates.append((0,learning_rate.get_value()))
previous_learning_rate = None   # type: typing.Union[float, None]
gradient_norms = []             # type: typing.List[float]
best_validation_loss = float("inf") # type: float
remaining_patience = config_params['patience']

logger.info(str(config_params))

for i in range(0, config_params['num_epochs']):
    epoch_start_time = time.time()
    
    train_iterator = preprocessing.get_generator(
        config_params['training_directory'], config_params['image_width'], config_params['batch_size'], type='training')
    threaded_train_iterator = preprocessing.get_threaded_generator(
        train_iterator, len(train_iterator.filenames),
        num_threads=config_params['num_threads_for_preprocessing'])
    avg_batch_train_loss = 0
    num_iterations = 0
    for images_labels in threaded_train_iterator:
        outputs, training_loss = train(numpy.moveaxis(images_labels[0],
                                                      3, 1),
                                       images_labels[1])
        avg_batch_train_loss += training_loss
        num_iterations += 1
        gradient_norms.append(get_gradient_norm(numpy.moveaxis(images_labels[0],
                                                               3, 1),
                                                images_labels[1]))
    avg_batch_train_loss = avg_batch_train_loss / num_iterations
    
    validation_iterator = preprocessing.get_generator(
        config_params['validation_directory'], config_params['image_width'], config_params['batch_size'], type='validation')
    threaded_validation_iterator = preprocessing.get_threaded_generator(
        validation_iterator, len(validation_iterator.filenames),
        num_threads=config_params['num_threads_for_preprocessing'])
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
    
    if avg_validate_loss < best_validation_loss:
        best_validation_loss = avg_validate_loss
        current_network_params = layers.get_all_params(network)
        best_network_params = layers.get_all_params(best_network)
        # Set best_network to current network parameters
        for i in range(len(best_network_params)):
            best_network_params[i].set_value(
                current_network_params[i].get_value())
        remaining_patience = config_params['patience']
    else:
        remaining_patience = remaining_patience - 1
    
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
    
    if remaining_patience == 0:
        logger.info('Best validation loss: ' + str(best_validation_loss) + '.')
        break
    
    if previous_learning_rate != learning_rate.get_value():
        logger.info('Changing learning rate from '
                    + str(previous_learning_rate) + ' to '
                    + str(learning_rate.get_value()) + '.')
        previous_learning_rates.append((i+1, learning_rate.get_value()))

logger.info('Average batch training loss per epoch: '
            + str(previous_validate_losses))
logger.info('Validation loss per epoch: '
            + str(previous_train_losses))
logger.info('Learning rate schedule: ' + str(previous_learning_rates))
logger.info('Gradient norms per iteration (not per epoch): ' + str(gradient_norms))

logger.info('Finishing run at ' + str(datetime.datetime.now()) + '.')
