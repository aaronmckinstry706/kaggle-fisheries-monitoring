import logging
import math
import unittest

import numpy
import scipy.misc as misc

import loader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class LoaderTest(unittest.TestCase):
    
    def readAndResizeImage(self, image_path):
        image = misc.imread(name=image_path, flatten=False)
        return misc.imresize(image, (self.resized_width, self.resized_width, 3))
    
    @classmethod
    def setUpClass(klass):
        super(LoaderTest, klass).setUpClass()
        logger = logging.getLogger(loader.__name__)
        logger.setLevel(logging.DEBUG)
        klass.stdoutLoggerHandler = logging.StreamHandler()
        logger.addHandler(klass.stdoutLoggerHandler)
    
    @classmethod
    def tearDownClass(klass):
        super(LoaderTest, klass).tearDownClass()
        logger = logging.getLogger(loader.__name__)
        logger.setLevel(logging.WARNING)
        logger.removeHandler(klass.stdoutLoggerHandler)
        del klass.stdoutLoggerHandler
    
    def setUp(self):
        self.num_train_images = 184 # This includes the validation images.
        self.num_validation_images = int(math.floor(184 * 0.3))
        self.num_test_images = 16
        self.batch_size = 3
        self.resized_width = 512
        self.data_loader = loader.Loader(
            self.batch_size,
            self.resized_width,
            'unit_test_resources/data/train',
            'unit_test_resources/data/test')
    
    def test_init(self):
        self.assertEqual(self.batch_size, self.data_loader._Loader__batch_size)
        self.assertEqual(
            self.resized_width, self.data_loader._Loader__resized_width)
        self.assertEqual('unit_test_resources/data/train',
                         self.data_loader._Loader__train_directory)
        self.assertEqual('unit_test_resources/data/test',
                         self.data_loader._Loader__test_directory)
        self.assertEqual(self.num_validation_images,
                         len(self.data_loader._Loader__validation_file_paths))
        self.assertEqual(self.num_train_images - self.num_validation_images,
                         len(self.data_loader._Loader__training_file_paths))
    
    def test_get_labels(self):
        labels = self.data_loader._get_labels()
        labels.sort()
        self.assertListEqual(
            labels,
            ["DOL", "LAG"])
    
    def test_get_relative_paths(self):
        relative_paths_alb = self.data_loader._get_relative_paths(
            'unit_test_resources/data/train')
        relative_paths_alb.sort()
        self.assertTrue('img_00165.jpg' in relative_paths_alb[0])
        self.assertTrue('img_00325.jpg' in relative_paths_alb[1])
        self.assertTrue('img_00348.jpg' in relative_paths_alb[2])
    
    def test_get_images(self):
        relative_paths = self.data_loader._get_relative_paths(
            'unit_test_resources/data/train')
        relative_paths.sort()
        self.assertEqual(184, len(relative_paths))
        images = self.data_loader._get_images(relative_paths[:4]
                                              + relative_paths[-4:])
        self.assertTupleEqual((8, self.resized_width, self.resized_width, 3),
                              images.shape)

    def test_get_images_with_labels(self):
        images_labels = self.data_loader._get_images_with_labels(
            self.data_loader._get_relative_paths(
                self.data_loader._Loader__train_directory))
        images = images_labels[0]
        labels = images_labels[1]
        
        expected_image = self.readAndResizeImage(
            'unit_test_resources/data/train/DOL/img_00165.jpg')
        
        # There are 184 total images in the unit_test_resources directory tree.
        self.assertTupleEqual(
            (self.num_train_images, self.resized_width, self.resized_width, 3),
            images.shape)
        self.assertTupleEqual((self.num_train_images, 8), labels.shape)
        self.assertTrue(numpy.array_equal(expected_image, images[0]))
        self.assertTrue(numpy.array_equal(numpy.array([1,0,0,0,0,0,0,0]),
                                          labels[0]))
        self.assertTrue(numpy.array_equal(numpy.array([0,1,0,0,0,0,0,0]),
                                          labels[-1]))
    
    def test_get_shuffled_train_input_iterator(self):
        training_iterator = self.data_loader.get_shuffled_train_input_iterator()
        num_iterations = 0
        expected_num_full_iterations = math.floor(
            (self.num_train_images - self.num_validation_images) \
            / self.batch_size)
        
        for images_labels in training_iterator:
            images = images_labels[0]
            labels = images_labels[1]
            
            current_batch_size = self.batch_size
            if num_iterations == expected_num_full_iterations:
                current_batch_size = (
                    self.num_train_images % self.batch_size)
            
            self.assertEqual(
                (current_batch_size, 3, self.resized_width, self.resized_width),
                images.shape)
            self.assertEqual((current_batch_size, 8), labels.shape)
            
            num_iterations = num_iterations + 1
        
        expected_num_iterations = math.ceil(
            float(self.num_train_images - self.num_validation_images) \
            / self.batch_size)
        self.assertEqual(expected_num_iterations, num_iterations)
    
    def test_get_validation_input_iterator(self):
        validation_iterator = self.data_loader.get_validation_input_iterator()
        num_iterations = 0
        expected_num_full_iterations = math.floor(
            self.num_validation_images / self.batch_size)
        
        for images_labels in validation_iterator:
            images = images_labels[0]
            labels = images_labels[1]
            
            current_batch_size = self.batch_size
            if num_iterations == expected_num_full_iterations:
                current_batch_size = (
                    self.num_validation_images % self.batch_size)
            
            self.assertEqual(
                (current_batch_size, 3, self.resized_width, self.resized_width),
                images.shape)
            self.assertEqual((current_batch_size, 8), labels.shape)
            
            num_iterations = num_iterations + 1
        
        expected_num_iterations = math.ceil(
            float(self.num_validation_images) / self.batch_size)
        self.assertEqual(expected_num_iterations, num_iterations)
    
    def test_get_test_input_iterator(self):
        test_iterator = self.data_loader.get_test_input_iterator()
        num_iterations = 0
        expected_num_full_iterations = math.floor(
            self.num_test_images / self.batch_size)
        
        for images in test_iterator:
            current_batch_size = self.batch_size
            if num_iterations == expected_num_full_iterations:
                current_batch_size = (
                    self.num_test_images % self.batch_size)
            
            self.assertEqual(
                (current_batch_size, 3, self.resized_width, self.resized_width),
                images.shape)
            
            num_iterations = num_iterations + 1
        
        expected_num_iterations = math.ceil(
            float(self.num_test_images) / self.batch_size)
        self.assertEqual(expected_num_iterations, num_iterations)

if __name__ == '__main__':
    unittest.main()
