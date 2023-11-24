import unittest
from src.datasets import PH2Dataset

class DatasetsCase(unittest.TestCase):

    def test_ph2_dataset(self):
        ds = PH2Dataset("data")
        img, mask = ds[0]
        self.assertEqual((765, 572), img.size)
        self.assertEqual(img.size, mask.size)


if __name__ == '__main__':
    unittest.main()

