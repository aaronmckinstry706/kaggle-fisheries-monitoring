import unittest

import loader

class LoaderTest(unittest.TestCase):
    
    def setUp(self):
        self.data_loader = loader.Loader(3, 512, 'data/train', 'data/test')
    
    def test_init(self):
        self.assertEqual(3, self.data_loader._Loader__batch_size)
        self.assertEqual(512, self.data_loader._Loader__resized_width)
        self.assertEqual('data/train', self.data_loader._Loader__train_directory)
        self.assertEqual('data/test', self.data_loader._Loader__test_directory)
    
    def test_get_labels(self):
        labels = self.data_loader._get_labels()
        labels.sort()
        self.assertListEqual(
            labels,
            ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"])
    
    def test_get_relative_paths(self):
        relative_paths_alb = self.data_loader._get_relative_paths(
                'data/train/ALB')
        relative_paths_alb.sort()
        self.assertTrue('img_00003.jpg' in relative_paths_alb[0])
        self.assertTrue('img_00010.jpg' in relative_paths_alb[1])
        self.assertTrue('img_00012.jpg' in relative_paths_alb[2])
    
    def test_get_images(self):
        pass
    
    def test_get_training_images_with_labels(self):
        pass
    
    def test_get_shuffled_train_input_iterator(self):
        pass
    
    def test_get_test_input_iterator(self):
        pass

if __name__ == '__main__':
    unittest.main()
