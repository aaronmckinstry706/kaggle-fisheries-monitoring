import unittest

import config

class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.unit_test_directory = 'unit_test_resources'
        self.config_file = self.unit_test_directory + '/config.txt'
        self.data_directory = self.unit_test_directory + '/data'
    
    def test_read_config_file(self):
        params = config.read_config_file(self.config_file)
        self.assertSetEqual(
            set([('training_directory', self.data_directory + '/train'),
                 ('validation_directory', self.data_directory + '/validation'),
                 ('test_directory', self.data_directory + '/test'),
                 ('num_threads_for_preprocessing', 8),
                 ('image_width', 512),
                 ('initial_learning_rate', 0.0001),
                 ('momentum', 0.9), 
                 ('batch_size', 64),
                 ('weight_decay', 0.0001),
                 ('patience', 40),
                 ('num_iterations', 40),
                 ('num_classes', 8)]),
            set(params.items()))

