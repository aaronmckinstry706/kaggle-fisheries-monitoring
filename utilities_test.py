import unittest
import utilities

class UtilitiesTest_PathReaders(unittest.TestCase):
    
    def test_get_labels(self):
        labels = utilities.get_labels('unit_test_resources/data/train')
        labels.sort()
        self.assertListEqual(
            labels,
            ["DOL", "LAG"])
    
    def test_get_relative_paths(self):
        relative_paths = utilities.get_relative_paths(
            'unit_test_resources/data/train')
        self.assertEqual(184, len(relative_paths))
        relative_paths.sort()
        self.assertTrue('img_00165.jpg' in relative_paths[0])
        self.assertTrue('img_00325.jpg' in relative_paths[1])
        self.assertTrue('img_00348.jpg' in relative_paths[2])

class UtilitiesTest_FileMovers(unittest.TestCase):
    
    def test00_separate_validation_set(self):
        utilities.separate_validation_set('unit_test_resources/data/train',
                                          'unit_test_resources/data/validation',
                                          1.0)
        self.assertEqual(
            0,
            len(utilities.get_relative_paths('unit_test_resources/data/train')))
        self.assertEqual(
            184,
            len(
                utilities.get_relative_paths(
                    'unit_test_resources/data/validation')))
    
    def test01_recombine_validation_and_training(self):
        utilities.recombine_validation_and_training(
            'unit_test_resources/data/validation',
            'unit_test_resources/data/train')
        self.assertEqual(
            184,
            len(utilities.get_relative_paths('unit_test_resources/data/train')))
        self.assertEqual(
            0,
            len(
                utilities.get_relative_paths(
                    'unit_test_resources/data/validation')))
