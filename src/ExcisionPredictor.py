import cv2
import numpy as np
import math
import imutils


def get_contour(mask):
    # For demo only
    img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_mask_contour(image, mask):
    contours = get_contour(mask)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image


class ExcisionPredictor(object):

    def __init__(self, qr_side_mm=5, work_width=256, margin=20, pix_in_mm=40):
        self.qr_side_mm = qr_side_mm
        self.work_width = work_width
        self.margin = margin
        self.pix_in_mm = pix_in_mm

    def get_margin(self, mask):
        """ TO DO
        :return: recommended excision margin in mm
        """
        return self.margin


    def get_scale(self, image_with_qr_code):
        qrCodeDetector = cv2.QRCodeDetector()
        decodedText, points, _ = qrCodeDetector.detectAndDecode(image_with_qr_code)
        if points is None:
            raise ValueError("QR Code not found")
        area = cv2.contourArea(points)
        pix_in_mm = math.sqrt(area) / self.qr_side_mm
        return pix_in_mm

    def resize(self, image, mask):
        init_width = mask.shape[0]
        scale_factor = init_width / self.work_width
        if scale_factor > 1:
            mask = imutils.resize(mask, width=self.work_width)
            image = imutils.resize(image, width=self.work_width)
            self.pix_in_mm = self.pix_in_mm / scale_factor
            pad = self.work_width  # // 2
            image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)))
            mask = np.pad(mask, ((pad, pad), (pad, pad), (0, 0)))

        return image, mask

    def dilate(self, mask):
        # https://en.wikipedia.org/wiki/Dilation_(morphology)
        margin = self.get_margin(mask)
        kernel_side = int(self.pix_in_mm * 2 * margin)
        kernel_shape = (kernel_side, kernel_side)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_shape)
        dilation = cv2.dilate(mask, kernel, iterations=1)
        return dilation




    def __call__(self, image, mask):
        if image.shape != mask.shape:
            raise ValueError("Mask and image must has equal shape")

        self.pix_in_mm = self.get_scale(image)
        image, mask = self.resize(image, mask)
        dilated_mask = self.dilate(mask)
        return dilated_mask

# image = cv2.imread(image_path)
