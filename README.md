# Melanoma_detection


Repository contain code for excision boundary prediction in class `ExcisionPredictor`.

To validate it prepare environment:

    python -m venv venv
    venv/bin/python -m ensurepip
    venv/bin/pip install -r requirements.txt
    
And run unittest:

    python -m unittest

Sample usage:
    
    from src.ExcisionPredictor import ExcisionPredictor
    import cv2

    image = cv2.imread("data/qr.jpg")
    mask = cv2.imread("data/mask.png")

        ep = ExcisionPredictor(
            qr_side_mm=5, # size of QR code on image
            work_width=256, # To speedup process image will be scaled to width=256
            margin=20, # Desired excision margin
            pix_in_mm=40 # default scale, for case without QR Code on image
        )
        image, dilated_mask = ep(
                                image, # image of skin leisson with QRCode 
                                mask # predicted earlier mask
                                )


### Main code in the colab notebooks:

[Classification](notebooks/classification.ipynb)\
[Segmentation](notebooks/segmentation.ipynb)
