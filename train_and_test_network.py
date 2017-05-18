import datetime
import logging
import os
import re
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
import matplotlib.pyplot as pyplot
import numpy
import theano
import theano.tensor as tensor

import config
import network_architectures as architectures
import preprocessing
import utilities

pyplot.ion()

now_str = re.sub(" ", "-", str(datetime.datetime.now()))
logger = logging.getLogger(__name__) # type: logging.Logger
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('logs/network_training_' + now_str + '.log'))
logger.setLevel(logging.INFO)
logger.info('\nStarting run at ' + now_str + '.')

config_params = config.read_config_file('config.txt')

# Define the symbolic variables and expressions for network computations.
learning_rate = theano.shared(numpy.float32(config_params['initial_learning_rate']))
inputs = tensor.tensor4(name='inputs')
labels = tensor.matrix(name='labels')

if (config_params['architecture'] == 'big'):
    network = architectures.fully_convolutional_big(
        inputs=inputs,
        image_shape=(3, config_params['image_width'],
                     config_params['image_width']),
        num_outputs=config_params['num_classes'])

    # This is used in early stopping. The parameters that achieve the lowest
    # error on the validation set are stored in this network.
    best_network = architectures.fully_convolutional_big(
        inputs=inputs,
        image_shape=(3, config_params['image_width'], config_params['image_width']),
        num_outputs=config_params['num_classes'])
elif (config_params['architecture'] == 'small'):
    network = architectures.fully_convolutional_small(
        inputs=inputs,
        image_shape=(3, config_params['image_width'],
                     config_params['image_width']),
        num_outputs=config_params['num_classes'])
    
    # This is used in early stopping. The parameters that achieve the lowest
    # error on the validation set are stored in this network.
    best_network = architectures.fully_convolutional_small(
        inputs=inputs,
        image_shape=(3, config_params['image_width'], config_params['image_width']),
        num_outputs=config_params['num_classes'])

training_outputs = layers.get_output(network, deterministic=False)
test_outputs = layers.get_output(network, deterministic=True)

network_parameters = layers.get_all_params(network, trainable=True)

training_loss_without_regularization = tensor.sum(
    objectives.categorical_crossentropy(
        training_outputs, labels)) / config_params['batch_size']
training_loss = training_loss_without_regularization \
    + lasagne.regularization.regularize_network_params(
        layer=network, penalty=regularization.l2)
validation_loss = objectives.categorical_crossentropy(test_outputs, labels)

gradients = tensor.concatenate(
    [tensor.flatten(tensor.grad(training_loss, params))
     for params in layers.get_all_params(network,
                                         trainable=True)])
gradient_norm = gradients.norm(2)
#hessian = tensor.hessian(training_loss, params_as_vector)

network_updates = updates.nesterov_momentum(
    loss_or_grads=training_loss,
    params=network_parameters,
    learning_rate=learning_rate,
    momentum=config_params['momentum'])

logger.info('Compiling the train and validation functions.')
train = theano.function(
    inputs=[inputs, labels],
    outputs=[training_outputs, training_loss],
    updates=network_updates)
validate = theano.function(inputs=[inputs, labels],
                           outputs=[test_outputs, validation_loss])
get_gradient_norm = theano.function(
    inputs=[inputs, labels],
    outputs=gradient_norm)

logger.info('Cleaning up training/validation split from previous runs.')
utilities.recombine_validation_and_training(config_params['validation_directory'],
                                            config_params['training_directory'])

logger.info('Splitting training and validation images.')
utilities.separate_validation_set(config_params['training_directory'], config_params['validation_directory'],
                                  split=0.1)

logger.info('Training model.')

previous_validation_losses = []   # type: typing.List[float]
previous_training_losses = []      # type: typing.List[float]
previous_learning_rates = []    # type: typing.List[typing.Tuple[int, float]]
previous_learning_rates.append(
    (0, numpy.asscalar(learning_rate.get_value())))
previous_learning_rate = None   # type: typing.Union[float, None]
gradient_norms = []             # type: typing.List[float]
best_validation_loss = float("inf") # type: float
remaining_patience = config_params['patience']
iteration_num = 0
epoch_num = 0

pyplot.figure(figsize=(6, 8))

logger.info(str(config_params))

while iteration_num < config_params['num_iterations']:
    epoch_start_time = time.time()
    
    training_iterator = preprocessing.get_generator(
        config_params['training_directory'], config_params['image_width'], config_params['batch_size'], type='training')
    threaded_training_iterator = preprocessing.get_threaded_generator(
        training_iterator, len(training_iterator.filenames),
        num_threads=config_params['num_threads_for_preprocessing'])
    for images_labels in threaded_training_iterator:
        outputs, current_training_loss = train(
            numpy.moveaxis(images_labels[0], 3, 1),
            images_labels[1])
        previous_training_losses.append(current_training_loss + 0.0)
        iteration_num += 1
        gradient_norms.append(
            numpy.asscalar(
                get_gradient_norm(
                    numpy.moveaxis(images_labels[0], 3, 1),
                    images_labels[1])))
        
        previous_learning_rate = learning_rate.get_value()
        learning_rate.set_value(
            numpy.float32(
                utilities.get_learning_rate(
                    previous_training_losses,
                    previous_validation_losses,
                    learning_rate.get_value())))
        
        if previous_learning_rate != learning_rate.get_value():
            logger.info('Changing learning rate from '
                        + str(numpy.asscalar(previous_learning_rate))
                        + ' to ' + str(numpy.asscalar(learning_rate.get_value()))
                        + '.')
            previous_learning_rates.append(
                (iteration_num + 1, numpy.asscalar(learning_rate.get_value())))
        
    
    validation_iterator = preprocessing.get_generator(
        config_params['validation_directory'], config_params['image_width'], config_params['batch_size'], type='validation')
    threaded_validation_iterator = preprocessing.get_threaded_generator(
        validation_iterator, len(validation_iterator.filenames),
        num_threads=config_params['num_threads_for_preprocessing'])
    avg_validation_loss = 0
    num_examples = 0
    for images_labels in threaded_validation_iterator:
        outputs, validation_loss = validate(numpy.moveaxis(images_labels[0],
                                                           3, 1),
                                            images_labels[1])
        avg_validation_loss += numpy.sum(validation_loss)
        num_examples += images_labels[0].shape[0]
    avg_validation_loss = avg_validation_loss / num_examples
    
    epoch_end_time = time.time()
    
    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        current_network_params = layers.get_all_params(network)
        best_network_params = layers.get_all_params(best_network)
        # Set best_network to current network parameters
        for j in range(len(best_network_params)):
            best_network_params[j].set_value(
                current_network_params[j].get_value())
        remaining_patience = config_params['patience']
    else:
        remaining_patience = remaining_patience - 1
    
    previous_validation_losses.append((iteration_num, avg_validation_loss))
    
    utilities.display_history(
        previous_training_losses,
        previous_validation_losses, 
        gradient_norms)
    
    logger.info('Epoch ' + str(epoch_num) + ':\n'
                + '    Crossentropy loss over validation set: '
                + str(avg_validation_loss) + '.\n'
                + '    Most recent batch loss over training set:'
                + ' ' + str(current_training_loss) + '.\n'
                + '    Time: '
                + str(int(round(epoch_end_time - epoch_start_time)))
                + ' seconds.')
    
    epoch_num = epoch_num + 1
    
    if remaining_patience == 0:
        logger.info('Best validation loss: ' + str(best_validation_loss) + '.')
        break
    
logger.info('Batch training loss per iteration: '
            + str(previous_training_losses))
logger.info('Validation loss after each epoch, indexed by iteration: '
            + str(previous_validation_losses))
logger.info('Gradient norms per iteration: ' + str(gradient_norms))
logger.info('Learning rate schedule: ' + str(previous_learning_rates))
logger.info('Finishing run at ' + str(datetime.datetime.now()) + '.')

pyplot.ioff()
pyplot.close('all')
