# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import os
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import imutils
from PIL import Image

#tessdata_dir_config = r'--tessdata-dir "/usr/local/Cellar/tesseract/3.05.01/share/tessdata"'
#pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.0/bin/tesseract'



class Classifier():
    def __init__(self):

        self.plot = False
        
        self.args = {
#                 "media" : im,           # type=str, help="path to input media")
                "type"  : "image",      # type=str, default="image", help="'video' or 'image'")
                "show"  : "original",    # type=str, default="edited", help="Show 'original' or 'edited'")

                "morph"  : "erosion",        # type=str, default="none", help="'erosion', 'dilation', 'opening' or 'closing'?")
                "morphH" : 3,           # type=int, default=5, help="Morphoplogy, Kernel height")
                "morphW" : 3,           # type=int, default=5, help="Morphoplogy, Kernel width")

                "blur"   :  "no",       # type=str, default="no", help="blur,'yes' or 'no'?")
                "blurH"  : 5,           # type=int, default=5, help="Blur, Kernel width")
                "blurW"  : 5,           # type=int, default=5, help="Blur, Kernel height")

                "targetW" : 600,        #type=int, default=600, help="nearest multiple of 32 for resized width, do not change")
                "targetH" : 500,        #type=int, default=500, help="nearest multiple of 32 for resized height, do not change")

                "binarization" : "yes", # type=str, default="yes", help="Apply binarization, 'yes' or 'no'")
                "bin1" : 105,           #type=int, default=140, help="Threshold 1")
                "bin2" : 255,           #type=int, default=255, help="Threshold 2")

                "greyscale" : "yes",    # type=str, default="yes", help="Apply greyscale, 'yes' or 'no'")
                "gR" : 0.299,           # type=float, default=0.299, help="Greyscale, R")
                "gG" : 0.587,           # type=float, default=0.587, help="Greyscale, G")
                "gB" : 0.114,           # type=float, default=0.114, help="Greyscale, B")

                "boxWL" : 70,          # type=int, default=130, help="Box width size, left")
                "boxWR" : 150,          # type=int, default=130, help="Box width size, right")

                "boxHT" : 120,           # type=int, default=35, help="Box height size, top")
                "boxHB" : 80,           # type=int, default=35, help="Box height size, bottom")

                # Tesserac OCR reader options
                "oem" : "2", # type=str, default="1", help="OEM SELECTION, model")
                # 0 = Original Tesseract only.
                # 1 = Neural nets LSTM only.
                # 2 = Tesseract + LSTM.
                # 3 = Default, based on what is available.

                "psm" : "3" #type=str, default="1",  help="PSM SELECTION, read line")
            }
        # 0 = Orientation and script detection (OSD) only.
        # 1 = Automatic page segmentation with OSD.
        # 2 = Automatic page segmentation, but no OSD, or OCR. (not implemented)
        # 3 = Fully automatic page segmentation, but no OSD. (Default)
        # 4 = Assume a single column of text of variable sizes.
        # 5 = Assume a single uniform block of vertically aligned text.
        # 6 = Assume a single uniform block of text.
        # 7 = Treat the image as a single text line.
        # 8 = Treat the image as a single word.
        # 9 = Treat the image as a single word in a circle.
        # 10 = Treat the image as a single character.
        # 11 = Sparse text. Find as much text as possible in no particular order.
        # 12 = Sparse text with OSD.
        # 13 = Raw line. Treat the image as a single text line,
        #      bypassing hacks that are Tesseract-specific.

        self.borderType = cv2.BORDER_CONSTANT

        # Used for testing
        # Desired sizes for image
        # targetW, targetH = 600, 500
        # boxWL, boxWR = 130, 130
        # boxHT, boxHB = 35, 25

        print(self.args)

    def predict(self, im):

        cv2.imwrite('input_image.jpg', im)
        image = cv2.imread('input_image.jpg')

        orig = image.copy()

        # Extract original sizes
        (origH, origW) = image.shape[:2]

        # Proportion of change
        rW = origW / float(self.args["targetW"])
        rH = origH / float(self.args["targetH"])

        # Resize image
        image_resized = cv2.resize(image, (self.args["targetW"], self.args["targetH"]))

        # Extract new sizes
        (H, W) = image_resized.shape[:2]

        # Decide extracted box
        startW, startH = int((W/2)-self.args["boxWL"]), int((H/2) - self.args["boxHT"])
        endW, endH = int((W/2)+self.args["boxWR"]), int((H/2) + self.args["boxHB"])

        # extract box
        crop_img = image_resized[startH:endH, startW:endW]

        crop_img = self.preprocessing(crop_img)
        
        # Pad image
        crop_img_padded = cv2.copyMakeBorder(crop_img, startH, int(H-endH), startW, int(W-endW), self.borderType, None, (255, 255, 255))

        # write box, on original image
        cv2.rectangle(image, (int(startW*rW), int(startH*rH)), (int(endW*rW), int(endH*rH)), (0, 232, 0), 5)

        # Config for OCR reader
        config = ("-l eng --oem " + str(self.args["oem"]) + " --psm " + str(self.args["psm"]))

        print('hejhejhej')
        text = pytesseract.image_to_string(crop_img_padded, config=config).encode("utf-8")
        print(text)
        
        return text

    def preprocessing(self, image):
        # Greysacle image
        if self.args["greyscale"] == "yes":
            image = self.greyscale([self.args["gR"], self.args["gG"], self.args["gB"]], image)

        # Binarization
        if self.args["binarization"] == "yes":
            ret, thresh1 = cv2.threshold(image, self.args["bin1"], self.args["bin2"], cv2.THRESH_BINARY)
            image = cv2.merge((thresh1, thresh1, thresh1))

        # th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
        # cv.THRESH_BINARY,11,2)
        # https://docs.opencv.org/3.4.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3

        # Morphology
        if self.args["morph"] != "none":
            image = self.morphology(image, iterations=1, flow=self.args["morph"], sizeW=self.args["morphW"], sizeH=self.args["morphH"])

        # Blurring
        if self.args["blur"] == "yes":
            image = cv2.blur(image, (self.args["blurH"], self.args["blurW"]))

        return image
        
    def greyscale(self, coefficients, image):
        coefficients = np.array(coefficients).reshape((1, 3))
        greyImage = cv2.transform(image, coefficients)
        return(greyImage)


    def morphology(self, image, flow, sizeW, sizeH, iterations=1):
        kernel = np.ones((sizeH, sizeW), np.uint8)
        if flow == "erosion":
            resultImg = cv2.erode(image, kernel, iterations=iterations)
        if flow == "dilation":
            resultImg = cv2.dilate(image, kernel, iterations=iterations)
        if flow == "opening":
            resultImg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        if flow == "closing":
            resultImg = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return(resultImg)




