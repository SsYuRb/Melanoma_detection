import unittest
from src.ExcisionPredictor import ExcisionPredictor, draw_mask_contour, get_contour
import cv2


class ExcisionPredictorTestCase(unittest.TestCase):
    def test_base(self):
        image = cv2.imread("data/qr.jpg")
        mask = cv2.imread("data/mask.png")

        ep = ExcisionPredictor()
        image, dilated_mask = ep(image, mask)
        contour = get_contour(dilated_mask)
        draw_mask_contour(image, dilated_mask)
        self.assertEqual(contour[0].shape[0], 926)
        #cv2.imwrite("test_base.png", image)

    def test_no_resize(self):
        image = cv2.imread("data/qr.jpg")
        mask = cv2.imread("data/mask.png")

        ep = ExcisionPredictor(work_width=None,  # disable rescaling
                               margin=2)  # for speedup
        image, dilated_mask = ep(image, mask)
        self.assertEqual(mask.shape, dilated_mask.shape)


if __name__ == '__main__':
    unittest.main()
