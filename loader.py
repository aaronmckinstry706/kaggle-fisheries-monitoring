"""
"""

import math
import os
import random

import numpy
import scipy.misc as misc

import logging

logger = logging.getLogger(__name__)

class Loader:

    def __init__(self, batchsize, resizedwidth, traindir, testdir):
        self.__batch_size = batchsize
        self.__resized_width = resizedwidth
        self.__train_directory = traindir
        self.__test_directory = testdir
        self.__training_file_paths = self._get_relative_paths(
            self.__train_directory)
        
        num_validation_images = int(math.floor(
            0.3 * len(self.__training_file_paths)))
        self.__validation_file_paths = random.sample(self.__training_file_paths,
                                                     num_validation_images)
        
        for path in self.__validation_file_paths:
            self.__training_file_paths.remove(path)
        
        self.training_data = None
        self.validation_data = None
        self.testing_data = None

    def _get_labels(self):
        """Gets a list of all directory names in the training directory. Each
        directory name is a label.
        
        Returns:
            A list of strings, each of which is a label.
        """
        labels = []
        for directory in os.listdir(self.__train_directory):
            labels.append(directory)
        labels.sort() # Just for deterministic behavior in unit tests.
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

    def _get_images(self, image_paths, images, offset):
        """
        """
        for i in range(0,len(image_paths)):
            images[i + offset] = misc.imresize(
                    misc.imread(image_paths[i]),
                    size=(self.__resized_width, self.__resized_width, 3))
        return images
    
    def _get_images_with_labels(self, paths):
        """
        """
        label_strings = self._get_labels()
        label_indexes = {
            label_strings[i] : i for i in range(0, len(label_strings))}
    
        all_images = numpy.zeros((len(paths), self.__resized_width,
                                  self.__resized_width, 3),
                                 dtype='float32')
        all_labels = numpy.zeros((len(paths), 8), dtype='float32')
        
        offset = 0
        for label_string in label_strings:
            image_paths_for_label = [
                path for path in paths
                    if label_string in path]
            image_paths_for_label.sort() # Just for deterministic unit tests.
            self._get_images(image_paths_for_label, all_images, offset)
            
            label_index = label_indexes[label_string]
            all_labels[
                offset:(offset + len(image_paths_for_label)), label_index] = 1.0
            
            offset += len(image_paths_for_label)
        
        return (all_images, all_labels)

    def _get_training_images_with_labels(self):
        return self._get_images_with_labels(self.__training_file_paths)
    
    def _get_validation_images_with_labels(self):
        return self._get_images_with_labels(self.__validation_file_paths)

    def get_shuffled_train_input_iterator(self):
        """
        """
        if self.training_data == None:
            (image_array, label_array) = self._get_training_images_with_labels()
            self.training_data = (image_array, label_array)
        else:
            (image_array, label_array) = self.training_data
        
        shuffled_indexes = numpy.arange(0, image_array.shape[0])
        num_iterations = int(math.ceil(float(
            image_array.shape[0]) / self.__batch_size))
        for i in range(0, num_iterations):
            current_batch_size = min(self.__batch_size,
                                     image_array.shape[0] - i*self.__batch_size)
            batch_start = i*self.__batch_size
            batch_end = i*self.__batch_size + current_batch_size
            batch_indexes = shuffled_indexes[batch_start:batch_end]
            yield (numpy.moveaxis(image_array[batch_indexes], 3, 1),
                   label_array[batch_indexes])
    
    def get_validation_input_iterator(self):
        """
        """
        if self.validation_data == None:
            (image_array, label_array) = self._get_validation_images_with_labels()
            self.validation_data = (image_array, label_array)
        else:
            (image_array, label_array) = self.validation_data
        
        num_iterations = int(math.ceil(float(
            image_array.shape[0]) / self.__batch_size))
        for i in range(0, num_iterations):
            current_batch_size = min(self.__batch_size,
                                     image_array.shape[0] - i*self.__batch_size)
            batch_start = i*self.__batch_size
            batch_end = i*self.__batch_size + current_batch_size
            yield (numpy.moveaxis(image_array[batch_start:batch_end], 3, 1),
                   label_array[batch_start:batch_end])

    def get_test_input_iterator(self):
        """
        """
        if self.testing_data == None:
            image_paths = self._get_relative_paths(self.__test_directory)
            image_array = numpy.zeros(
                (len(image_paths), self.__resized_width, self.__resized_width, 3),
                dtype='float32')
            self._get_images(image_paths, image_array, offset=0)
            self.testing_data = image_array
        else:
            image_array = self.testing_data
    
        shuffled_indexes = numpy.arange(0, image_array.shape[0])
        num_iterations = int(math.ceil(
            float(image_array.shape[0])/self.__batch_size))
        for i in range(0, num_iterations):
            current_batch_size = min(self.__batch_size,
                                     image_array.shape[0] - i*self.__batch_size)
            batch_start = i*self.__batch_size
            batch_end = i*self.__batch_size + current_batch_size
            batch_indexes = shuffled_indexes[batch_start:batch_end]
            yield numpy.moveaxis(image_array[batch_indexes], 3, 1)


