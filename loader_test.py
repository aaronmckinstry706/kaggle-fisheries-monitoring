import logging
import unittest

import numpy
import scipy.misc as misc

import loader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class LoaderTest(unittest.TestCase):
    
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
        self.data_loader = loader.Loader(
            3,
            512,
            'unit_test_resources/data/train',
            'unit_test_resources/data/test')
    
    def test_init(self):
        self.assertEqual(3, self.data_loader._Loader__batch_size)
        self.assertEqual(512, self.data_loader._Loader__resized_width)
        self.assertEqual('unit_test_resources/data/train',
                         self.data_loader._Loader__train_directory)
        self.assertEqual('unit_test_resources/data/test',
                         self.data_loader._Loader__test_directory)
    
    def test_get_labels(self):
        labels = self.data_loader._get_labels()
        labels.sort()
        self.assertListEqual(
            labels,
            ["LAG"])
    
    def test_get_relative_paths(self):
        relative_paths_alb = self.data_loader._get_relative_paths(
            'unit_test_resources/data/train/LAG')
        relative_paths_alb.sort()
        self.assertTrue('img_00091.jpg' in relative_paths_alb[0])
        self.assertTrue('img_00176.jpg' in relative_paths_alb[1])
        self.assertTrue('img_00657.jpg' in relative_paths_alb[2])
    
    def test_get_images(self):
        relative_paths = self.data_loader._get_relative_paths(
            'unit_test_resources/data/train')
        relative_paths.sort()
        images = self.data_loader._get_images(relative_paths[0:4])
        self.assertTupleEqual((4, 512, 512, 3), images.shape)

    def test_get_training_images_with_labels(self):
        images_labels = self.data_loader._get_training_images_with_labels()
        images = images_labels[0]
        labels = images_labels[1]
        
        image = misc.imread(
            name='unit_test_resources/data/train/LAG/img_00091.jpg',
            flatten=False)
        resized_image = misc.imresize(image, (512, 512, 3))
        
        self.assertTupleEqual((67, 512, 512, 3), images.shape)
        self.assertTupleEqual((67, 8), labels.shape)
        self.assertTrue(numpy.array_equal(resized_image, images[0]))
        self.assertTrue(numpy.array_equal(numpy.array([1,0,0,0,0,0,0,0]),
                                          labels[0]))
    
    def test_get_shuffled_train_input_iterator(self):
        pass
    
    def test_get_test_input_iterator(self):
        pass

if __name__ == '__main__':
    unittest.main()
