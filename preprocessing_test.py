import unittest

import scipy.misc as misc

import preprocessing
import utilities

class PreprocessingTest(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.training_directory = 'unit_test_resources/data/train'
        self.test_directory = 'unit_test_resources/data/test'
        self.image_width = 512
        self.batch_size = 4
        self.num_labels = len(utilities.get_labels(self.training_directory))
    
    def test_resize_and_random_crop(self):
        resize = preprocessing.resize_and_random_crop(self.image_width)
        resized_image = resize(
            misc.imread(self.training_directory + '/DOL/img_00165.jpg'))
        self.assertTrue(max(resized_image.shape[:2]) >= self.image_width)
        self.assertEqual(self.image_width, min(resized_image.shape[:2]))
    
    def test_resize_and_center_crop(self):
        resize = preprocessing.resize_and_center_crop(self.image_width)
        resized_image = resize(
            misc.imread(self.training_directory + '/DOL/img_00165.jpg'))
        self.assertTrue(max(resized_image.shape[:2]) >= self.image_width)
        self.assertEqual(self.image_width, min(resized_image.shape[:2]))

    def test_get_training_generator(self):
        training_generator = preprocessing.get_training_generator(
            self.training_directory, self.image_width, self.batch_size)
        batches = 0
        for images, labels in training_generator:
            self.assertTupleEqual((self.batch_size, self.image_width,
                                   self.image_width, 3), images.shape)
            self.assertTupleEqual((self.batch_size, self.num_labels),
                                  labels.shape)
            batches += 1
            if batches == 30:
                break
    
    def test_get_validation_generator(self):
        validation_generator = preprocessing.get_validation_generator(
            self.training_directory, self.image_width, self.batch_size)
        batches = 0
        for images, labels in validation_generator:
            self.assertTupleEqual((self.batch_size, self.image_width,
                                   self.image_width, 3), images.shape)
            self.assertTupleEqual((self.batch_size, self.num_labels),
                                  labels.shape)
            batches += 1
            if batches == 30:
                break
    
    def test_get_test_generator(self):
        test_generator = preprocessing.get_test_generator(
            self.test_directory, self.image_width, self.batch_size)
        batches = 0
        for images in test_generator:
            self.assertTupleEqual((self.batch_size, self.image_width,
                                   self.image_width, 3), images.shape)
            batches += 1
            if batches == 30:
                break
    
    def test_get_threaded_generator__one_thread(self):
        training_generator = preprocessing.get_threaded_generator(
            self.training_directory,
            self.image_width,
            self.batch_size,
            type='training',
            num_threads=1)
        for images, labels in training_generator:
            self.assertTupleEqual((self.batch_size, self.image_width,
                                   self.image_width, 3), images.shape)
            self.assertTupleEqual((self.batch_size, self.num_labels),
                                  labels.shape)

    def test_get_threaded_generator__eight_threads(self):
        training_generator = preprocessing.get_threaded_generator(
            self.training_directory,
            self.image_width,
            self.batch_size,
            type='training',
            num_threads=8)
        for images, labels in training_generator:
            self.assertTupleEqual((self.batch_size, self.image_width,
                                   self.image_width, 3), images.shape)
            self.assertTupleEqual((self.batch_size, self.num_labels),
                                  labels.shape)

