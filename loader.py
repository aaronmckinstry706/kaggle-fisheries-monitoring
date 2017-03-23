"""A module containing all methods for loading and iterating through the data. 
"""

import os
import math

import numpy
import scipy.misc as misc

class Loader:

    def __init__(self, batchsize, resizedwidth, traindir, testdir):
        self.__batch_size = batchsize
        self.__resized_width = resizedwidth
        self.__train_directory = traindir
        self.__test_directory = testdir

    def _get_labels(self):
        """Gets a list of all directory names in the training directory. Each
        directory name is a label.
        
        Returns:
            A list of strings, each of which is a label.
        """
        labels = []
        for directory in os.listdir(self.__train_directory):
            labels.append(directory)
        return labels

    def _get_relative_paths(self, root_directory):
        """Gets a list of all path names in root_directory that have extension
        '.jpg'. Any returned path will be relative to the current working
        directory.

        Args:
            root_directory -- The path for the directory that will be searched.
            file_extension -- A string containing a file extension used to
                              determine which files to include.

        Returns:
            A list of file paths, each relative to the current directory ('.').
            Every file in root_directory (or its subdirectories) that ends in
            file_extension is included in this list; no other files are
            included.
        """
        file_extension = '.jpg'
        relative_file_paths = []
        for directory, _, file_names in os.walk(root_directory):
            valid_file_names = (
                    name for name in file_names
                    if name[-len(file_extension):] == file_extension)
            valid_file_paths = (os.path.relpath(directory + '/' + name, '.')
                                for name in valid_file_names)
            relative_file_paths.extend(valid_file_paths)
        return relative_file_paths

    def _get_images(self, image_paths, resized_width):
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

    def _get_training_images_with_labels(self, resized_width):
        """Given the final width of the training images, this returns a tuple
        (images, labels). images[i] is a training image, and labels[i] is the label
        for that image. images will have shape (n, resized_width, resized_width, 3).
        labels will have shape (n, 8). 
    
        Args:
            resized_width -- The desired width of training images.
    
        Returns:
            (images, labels) -- The tuple of all images and corresponding labels in
                                the training set. 
        """
        label_strings = get_labels()
        label_indexes = {label_strings[i] : i for i in range(0, len(label_strings))}
    
        all_images = numpy.zeros((0, resized_width, resized_width, 3))
        all_labels = numpy.zeros((0, 8))
    
        for label_string in label_strings:
            image_paths_for_label = get_relative_paths(
                    TRAINING_DIRECTORY + "/" + label_string, '.jpg')
            images_for_label = get_images(image_paths_for_label, resized_width)
            labels = numpy.zeros((images_for_label.shape[0], 8))
            label_index = label_indexes[label_string]
            labels[:, label_index] = 1
    
            numpy.append(all_images, images_for_label)
            numpy.append(all_labels, labels)
    
        return (all_images, all_labels)

    def get_shuffled_train_input_iterator(self, batch_size, resized_width):
        """Given an ndarray of images with shape (a, b, c, 3), this returns an
        iterator which provides ndarray images of shape (batch_size, 3, b, c)
        with the elements rearranged accordingly.
        """
        (image_array, label_array) = get_training_images_with_labels(resized_width)
        shuffled_indexes = numpy.arange(0, image_array.shape[0])
        num_iterations = int(math.ceil(float(image_array.shape[0])/batch_size))
        print(num_iterations)
        for i in range(0, num_iterations):
            current_batch_size = min(batch_size,
                                     image_array.shape[0] - i*batch_size)
            batch_start = i*batch_size
            batch_end = i*batch_size + current_batch_size
            batch_indexes = shuffled_indexes[batch_start:batch_end]
            yield (numpy.moveaxis(image_array[batch_indexes], 3, 1),
                   label_array[batch_indexes])

    def get_test_input_iterator(self, batch_size, resized_width):
        """Given an ndarray of images with shape (a, b, c, 3), this returns an
        iterator which provides ndarray images of shape (batch_size, 3, b, c)
        with the elements rearranged accordingly.
        """
        image_paths = get_relative_paths(TEST_DIRECTORY, '.jpg')
        image_array = get_images(test_image_paths, resized_width)
    
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


