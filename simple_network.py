# All variables and functions are named according to machine learning
# terminology: example, input, output, label, etc. For example, an input could be
# a scalar, or a vector, or a matrix, or a tensor.

import os
import math

import lasagne
import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import lasagne.objectives as objectives
import lasagne.updates as updates
import numpy
import scipy.misc as misc
import theano
import theano.tensor as tensor

def get_relative_paths(root_directory, file_extension):
    """Gets a list of all path names in root_directory that have extension
    file_extension. Any returned path will be relative to the current working
    directory.

    Args:
        root_directory -- The path for the directory that will be searched.
        file_extension -- A string containing a file extension used to determine
                          which files to include.

    Returns:
        A list of file paths, each relative to the current directory ('.').
        Every file in root_directory (or its subdirectories) that ends in
        file_extension is included in this list; no other files are included.
    """
    relative_file_paths = []
    for directory, _, file_names in os.walk(root_directory):
        valid_file_names = (name for name in file_names
                            if name[-len(file_extension):] == file_extension)
        valid_file_paths = (os.path.relpath(directory + '/' + name, '.')
                            for name in valid_file_names)
        relative_file_paths.extend(valid_file_paths)
    return relative_file_paths

def get_images(image_paths, resized_width):
    """Given a list of image paths (absolute or relative), and a desired image
    width, this returns a numpy array with the shape (len(image_paths),
    resized_width, resized_width, 3). If images is returned, then images[i] is
    the ndarray of the image at image_paths[i].

    Args:
        image_paths -- A list of image paths. Each image whose file is listed
                       should be in JPEG format with RGB color channels.

    Returns:
        images -- A numpy.ndarray of all the images in image_paths.
    """
    images = numpy.empty(
        shape=(len(image_paths), resized_width, resized_width, 3),
        dtype='float32')
    for i in range(0,len(image_paths)):
        images[i] = misc.imresize(
            misc.imread(image_paths[i]), size=(resized_width, resized_width, 3))
    return images

def get_shuffled_input_iterator(image_array, batch_size):
    """Given an ndarray of images with shape (a, b, c, 3), this returns an
    iterator which provides ndarray images of shape (batch_size, 3, b, c)
    with the elements rearranged accordingly.
    """
    shuffled_indexes = numpy.arange(0, image_array.shape[0])
    num_iterations = int(math.ceil(float(image_array.shape[0])/batch_size))
    print(num_iterations)
    for i in range(0, num_iterations):
        current_batch_size = min(batch_size,
                                 image_array.shape[0] - i*batch_size)
        batch_start = i*batch_size
        batch_end = i*batch_size + current_batch_size
        batch_indexes = shuffled_indexes[batch_start:batch_end]
        yield numpy.moveaxis(image_array[batch_indexes], 3, 1)

# Set the preprocessing parameters.
image_width = 512

# Set the learning algorithm parameters.
learning_rate = 1.0
momentum = 0.9
batch_size = 64

# Set some architectural parameter invariants
num_classes = 8

# Define the symbolic variables and expressions for network computation.
inputs = tensor.tensor3(name='inputs')
labels = tensor.matrix(name='labels')

input_layer = layers.InputLayer(shape=(None, image_width, image_width),
                                input_var=inputs, name='input_layer')
input_layer_reshaped = layers.ReshapeLayer(incoming=input_layer, shape=(64*64,),
                                           name='input_layer_reshaped')
output_layer = layers.DenseLayer(
    incoming=input_layer_reshaped, num_units=8,
    nonlinearity=nonlinearities.softmax, name='output_layer')

network_parameters = layers.get_all_params(output_layer)
outputs = layers.get_output(output_layer)
unaveraged_losses = tensor.sum(tensor.sum(objectives.squared_error(outputs,
                                                                   labels)))
loss = tensor.sum(unaveraged_losses) / (batch_size*8)

# Compile the train, test, and validation functions
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
