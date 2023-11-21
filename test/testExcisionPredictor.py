import unittest
from src.ExcisionPredictor import ExcisionPredictor, draw_mask_contour
import cv2

class ExcisionPredictorTestCase(unittest.TestCase):
    def test_base(self):

        ep = ExcisionPredictor()
        image = cv2.imread("data/qr.jpg")
        mask = cv2.imread("data/mask.png")
        dilated_mask = ep(image,mask)
        draw_mask_contour(image,dilated_mask)
        cv2.imwrite("test_base.png", image)
        #self.assertGreater(len(contours) ,0)


if __name__ == '__main__':
    unittest.main()
