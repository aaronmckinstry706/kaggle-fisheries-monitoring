import unittest

import scipy.misc as misc

import preprocessing

class PreprocessingTest(unittest.TestCase):
    
    def test_resize_and_random_crop(self):
        resize = preprocessing.resize_and_random_crop(512)
        resized_image = resize(
            misc.imread('unit_test_resources/data/train/DOL/img_00165.jpg'))
        self.assertTrue(max(resized_image.shape[:2]) >= 512)
        self.assertEqual(512, min(resized_image.shape[:2]))
    
    def test_resize_and_center_crop(self):
        resize = preprocessing.resize_and_center_crop(512)
        resized_image = resize(
            misc.imread('unit_test_resources/data/train/DOL/img_00165.jpg'))
        self.assertTrue(max(resized_image.shape[:2]) >= 512)
        self.assertEqual(512, min(resized_image.shape[:2]))
