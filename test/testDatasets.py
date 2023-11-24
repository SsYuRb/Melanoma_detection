import unittest
from src.datasets import PH2Dataset, ISICDataset

class DatasetsCase(unittest.TestCase):


    def test_ph2_dataset(self):
        ds = PH2Dataset("data")
        img, mask = ds[0]
        self.assertEqual(200, len(ds))
        self.assertEqual((765, 572), img.size)
        self.assertEqual(img.size, mask.size)
        img = None
        mask = None

    def test_isic_dataset(self):
        ds = ISICDataset("data")
        print(len(ds))
        img, mask, label = ds[0]
        self.assertEqual(0, label)
        self.assertEqual((1022, 767), img.size)
        self.assertEqual(img.size, mask.size)


if __name__ == '__main__':
    unittest.main()

