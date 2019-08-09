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

def greyscale(coefficients, image):
    coefficients = np.array(coefficients).reshape((1, 3))
    greyImage = cv2.transform(image, coefficients)
    return(greyImage)


def morphology(image, flow, sizeW, sizeH, iterations=1):
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


# Example use
# python preProcessing.py -m C:\Users\Augus\PycharmProjects\scanner\videos\arla_film.mp4 -t video -show original -morph erosion -morphH 5 -morphW 5 -blur yes -morphH 5 -morphW 5
# python preProcessing.py -m C:\Users\Augus\PycharmProjects\scanner\videos\images\arlamjolk2.png -t video -show original -morph erosion -morphH 5 -morphW 5 -blur yes -morphH 5 -morphW 5
# python preProcessing.py -m C:\Users\Augus\PycharmProjects\scanner\images\mjolkny.jpg -t image -show original -bwl 70 -bwr 15 -bht 0 -bhb 40 -binarization yes -greyscale yes -b1 145 -b2 255 -morph erosion -morphH 3 -morphW 3 -blur no -oem 2 -psm 5
# python preProcessing.py -m C:\Users\Augus\PycharmProjects\scanner\images\mjolkny.jpg -t image -show edited -bwl 70 -bwr 15 -bht 0 -bhb 40 -binarization yes -greyscale yes -b1 145 -b2 255 -morph erosion -morphH 3 -morphW 3 -blur no -oem 2 -psm 3



# Parser arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--media", type=str, help="path to input media")
ap.add_argument("-t", "--type", type=str, default="image", help="'video' or 'image'")
ap.add_argument("-show", "--show", type=str, default="edited", help="Show 'original' or 'edited'")

# Morphing with options
ap.add_argument("-morph", "--morph", type=str, default="none", help="'erosion', 'dilation', 'opening' or 'closing'?")
ap.add_argument("-morphH", "--morphH", type=int, default=5, help="Morphoplogy, Kernel height")
ap.add_argument("-morphW", "--morphW", type=int, default=5, help="Morphoplogy, Kernel width")

# Blurring with options
ap.add_argument("-blur", "--blur", type=str, default="no", help="blur,'yes' or 'no'?")
ap.add_argument("-blurH", "--blurH", type=int, default=5, help="Blur, Kernel width")
ap.add_argument("-blurW", "--blurW", type=int, default=5, help="Blur, Kernel height")

# Target image size
ap.add_argument("-imgW", "--targetW", type=int, default=600, help="nearest multiple of 32 for resized width, do not change")
ap.add_argument("-imgH", "--targetH", type=int, default=500, help="nearest multiple of 32 for resized height, do not change")

# Binarization
ap.add_argument("-binarization", "--binarization", type=str, default="yes", help="Apply binarization, 'yes' or 'no'")
ap.add_argument("-b1", "--bin1", type=int, default=140, help="Threshold 1")
ap.add_argument("-b2", "--bin2", type=int, default=255, help="Threshold 2")

# Greyscaling
ap.add_argument("-greyscale", "--greyscale", type=str, default="yes", help="Apply greyscale, 'yes' or 'no'")
ap.add_argument("-gR", "--gR", type=float, default=0.299, help="Greyscale, R")
ap.add_argument("-gG", "--gG", type=float, default=0.587, help="Greyscale, G")
ap.add_argument("-gB", "--gB", type=float, default=0.114, help="Greyscale, B")

# Box selection
ap.add_argument("-bwl", "--boxWL", type=int, default=130,
                help="Box width size, left")
ap.add_argument("-bwr", "--boxWR", type=int, default=130,
                help="Box width size, right")

ap.add_argument("-bht", "--boxHT", type=int, default=35,
                help="Box height size, top")
ap.add_argument("-bhb", "--boxHB", type=int, default=35,
                help="Box height size, bottom")

# Tesserac OCR reader options
ap.add_argument("-oem", "--oem", type=str, default="1",
                help="OEM SELECTION, model")
# 0 = Original Tesseract only.
# 1 = Neural nets LSTM only.
# 2 = Tesseract + LSTM.
# 3 = Default, based on what is available.

ap.add_argument("-psm", "--psm", type=str, default="1",
                help="PSM SELECTION, read line")
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

args = vars(ap.parse_args())
borderType = cv2.BORDER_CONSTANT

# Used for testing
# Desired sizes for image
# targetW, targetH = 600, 500
# boxWL, boxWR = 130, 130
# boxHT, boxHB = 35, 25



if args["type"] == "video":
    # Set up the video stream
    vs = cv2.VideoCapture(args["media"])

    # start the FPS throughput estimator
    fps = FPS().start()

    # loop over frames from the video stream
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1] if args.get("media", False) else frame

        # check to see if we have reached the end of the stream
        if frame is None:
            break

        # resize the frame, maintaining the aspect ratio
        frame = cv2.rotate(frame, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        frame = imutils.resize(frame, width=600)
        orig=frame.copy()
        # Extract original sizes
        (origH, origW) = frame.shape[:2]

        # Proportion of change
        rW = origW / float(args["targetW"])
        rH = origH / float(args["targetH"])

        # Resize image
        frame_resized = cv2.resize(frame, (args["targetW"], args["targetH"]))

        # Extract new sizes
        (H, W) = frame_resized.shape[:2]

        # Decide extracted box
        startW, startH = int((W / 2) - args["boxWL"]), int((H / 2) - args["boxHT"])
        endW, endH = int((W / 2) + args["boxWR"]), int((H / 2) + args["boxHB"])

        # extract box
        crop_img = frame_resized[startH:endH, startW:endW]

        # Greysacle image
        if args["greyscale"] == "yes":
            crop_img = greyscale([args["gR"], args["gG"],args["gB"]], crop_img)

        # Binarization
        if args["binarization"] == "yes":
            ret, thresh1 = cv2.threshold(crop_img, args["bin1"], args["bin2"], cv2.THRESH_BINARY)
            crop_img = cv2.merge((thresh1, thresh1, thresh1))

        # Morphology
        if args["morph"] != "none":
            crop_img = morphology(crop_img, iterations=1, flow=args["morph"], sizeW=args["morphW"] , sizeH=args["morphH"])

        # Blurring
        if args["blur"] == "yes":
            crop_img = cv2.blur(crop_img, (args["blurH"], args["blurW"]))

        # Pad image
        crop_img_padded = cv2.copyMakeBorder(crop_img, startH, int(H - endH), startW, int(W - endW), borderType,
                                             None, (255, 255, 255))

        # write box, on original image
        cv2.rectangle(orig, (int(startW * rW), int(startH * rH)), (int(endW * rW), int(endH * rH)), (0, 232, 0), 2)
        # Config for OCR reader
        # config = ("-l eng --oem " + args["oem"] + " --psm " + args["psm"] + " -c tessedit_char_whitelist=0123456789")
        config = ("-l eng --oem " + str(args["oem"]) + " --psm " + str(args["psm"]))

        # text = pytesseract.image_to_string(roi, config=config)
        text = pytesseract.image_to_string(crop_img_padded, config=config)
        # testText = "lol"

        # Put text on image
        cv2.putText(orig, text, (int(startW * rW), int(startH * rH) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if args["show"] == "original":
            cv2.imshow("Text detection", orig)
        if args["show"] == "edited":
            cv2.imshow("Text detection", crop_img_padded)

        key = cv2.waitKey(1) & 0xFF

        # display the text OCR'd by Tesseract
        #print("OCR TEXT")
        #print("========")
        print(text)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # otherwise, release the file pointer
    vs.release()

    # close all windows
    cv2.destroyAllWindows()

else:
    image = cv2.imread(args["media"])
    orig = image.copy()

    # Extract original sizes
    (origH, origW) = image.shape[:2]

    # Proportion of change
    rW = origW / float(args["targetW"])
    rH = origH / float(args["targetH"])

    # Resize image
    image_resized = cv2.resize(image, (args["targetW"], args["targetH"]))

    # Extract new sizes
    (H, W) = image_resized.shape[:2]

    # Decide extracted box
    startW, startH = int((W/2)-args["boxWL"]), int((H/2)-args["boxHT"])
    endW, endH = int((W/2)+args["boxWR"]), int((H/2)+args["boxHB"])

    # extract box
    crop_img = image_resized[startH:endH, startW:endW]

    # Greysacle image
    if args["greyscale"] == "yes":
        crop_img = greyscale([args["gR"], args["gG"], args["gB"]], crop_img)

    # Binarization
    if args["binarization"] == "yes":
        ret, thresh1 = cv2.threshold(crop_img, args["bin1"], args["bin2"], cv2.THRESH_BINARY)
        crop_img = cv2.merge((thresh1, thresh1, thresh1))

    # Morphology
    if args["morph"] != "none":
        crop_img = morphology(crop_img, iterations=1, flow=args["morph"], sizeW=args["morphW"], sizeH=args["morphH"])

    # Blurring
    if args["blur"] == "yes":
        crop_img = cv2.blur(crop_img, (args["blurH"], args["blurW"]))

    # Pad image
    crop_img_padded = cv2.copyMakeBorder(crop_img, startH, int(H-endH), startW, int(W-endW), borderType, None, (255, 255, 255))

    # write box, on original image
    cv2.rectangle(image, (int(startW*rW), int(startH*rH)), (int(endW*rW), int(endH*rH)), (0, 232, 0), 5)

    # Config for OCR reader
    # config = ("-l eng --oem " + args["oem"] + " --psm " + args["psm"] + " -c tessedit_char_whitelist=0123456789")
    config = ("-l eng --oem " + str(args["oem"]) + " --psm " + str(args["psm"]))

    # text = pytesseract.image_to_string(roi, config=config)
    text = pytesseract.image_to_string(crop_img_padded, config=config)
    # testText = "test"

    # Put text on image
    cv2.putText(image, text, (int(startW*rW), int(startH*rH) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 3)


    # display the text OCR'd by Tesseract
    print("OCR TEXT")
    print("========")
    print(text)

    # show the output image
    # cv2.imshow("car_wash", crop_img)
    # cv2.imshow("car_wash", image_resized)
    # cv2.imshow("car_wash", crop_img_padded)
    # cv2.imshow("car_wash", image)
    # cv2.imshow("car_wash", binaryImage)

    if args["show"] == "original":
        (H, W) = image.shape[:2]
        smaller = cv2.resize(image, (int(round(W / 5)), int(round(H / 5))))
        cv2.imshow("Text detection", smaller)
        cv2.waitKey(0)
    if args["show"] == "edited":
        cv2.imshow("Text detection", crop_img_padded)
        cv2.waitKey(0)




































