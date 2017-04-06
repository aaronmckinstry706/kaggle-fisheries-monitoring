import unittest
import utilities

import os

class UtilitiesTest_PathReaders(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.training_directory = 'unit_test_resources/data/train'
    
    def test_get_labels(self):
        labels = utilities.get_labels(self.training_directory)
        labels.sort()
        self.assertListEqual(
            labels,
            ["DOL", "LAG"])
    
    def test_get_relative_paths(self):
        relative_paths = utilities.get_relative_paths(
            self.training_directory)
        self.assertEqual(184, len(relative_paths))
        relative_paths.sort()
        self.assertTrue('img_00165.jpg' in relative_paths[0])
        self.assertTrue('img_00325.jpg' in relative_paths[1])
        self.assertTrue('img_00348.jpg' in relative_paths[2])

class UtilitiesTest_FileMovers(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.training_directory = 'unit_test_resources/data/train'
        self.validation_directory = 'unit_test_resources/data/validation'
    
    def test00_separate_validation_set_splitOne(self):
        utilities.separate_validation_set(self.training_directory,
                                          self.validation_directory,
                                          1.0)
        self.assertEqual(
            0,
            len(utilities.get_relative_paths(self.training_directory)))
        self.assertEqual(
            184,
            len(
                utilities.get_relative_paths(
                    self.validation_directory)))
    
    def test01_recombine_validation_and_training_splitOne(self):
        utilities.recombine_validation_and_training(
            self.validation_directory,
            self.training_directory)
        self.assertEqual(
            184,
            len(utilities.get_relative_paths(self.training_directory)))
        self.assertEqual(
            0,
            len(
                utilities.get_relative_paths(
                    self.validation_directory)))
    
    def test10_separate_validation_set_splitQuarter(self):
        utilities.separate_validation_set(self.training_directory,
                                          self.validation_directory,
                                          split=0.25)
        self.assertEqual(
            139,
            len(utilities.get_relative_paths(self.training_directory)))
        self.assertEqual(
            45,
            len(
                utilities.get_relative_paths(
                    self.validation_directory)))
    
    def test11_recombine_validation_and_training_splitQuarter(self):
        utilities.recombine_validation_and_training(
            self.validation_directory,
            self.training_directory)
        self.assertEqual(
            184,
            len(utilities.get_relative_paths(self.training_directory)))
        self.assertEqual(
            0,
            len(
                utilities.get_relative_paths(
                    self.validation_directory)))

