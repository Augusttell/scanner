# TODO Deal with the multiple boxes per frame.
# TODO Add conditional, if text = detected -> perform OCR, add threshold as variable  - done, needs to be tested TODO

# TODO Add and test gaussian thresholding
# TODO Fix breaking iside of loop

# TODO Add main function
# TODO Write proper Readme
# TODO add variable for greyscale of video detection algo?

### Example use / test cases ###
## Test case 1, Video with Detection, works
# python preProcessing.py -m C:\Users\Augus\PycharmProjects\scanner\videos\arla_film.mp4 -t video -show original -morph erosion -morphH 3 -morphW 3 -binarization yes -b1 140 -detection detection -tthresh 2 -dthresh 0.9  -bwl 70 -bwr 15 -bht 0 -bhb 40 -paddingH 1.05 -paddingW 1.05 -modelLoc C:\Users\Augus\PycharmProjects\scanner\opencv-text-recognition\frozen_east_text_detection.pb -oem 2 -psm 3

## Test case 2, Video with box, works
# python preProcessing.py -m C:\Users\Augus\PycharmProjects\scanner\videos\arla_film.mp4 -t video -show original -morph erosion -morphH 3 -morphW 3 -binarization yes -b1 140 -detection pre-defined -tthresh 2 -dthresh 0.9  -bwl 70 -bwr 15 -bht 0 -bhb 40 -oem 2 -psm 3

## Test case 3, Image with detection, works
# python preProcessing.py -m C:\Users\Augus\PycharmProjects\scanner\images\mjolkny.jpg -t image -show original -morph erosion -morphH 3 -morphW 3 -binarization yes -b1 140 -detection detection -tthresh 2 -dthresh 0.9 -paddingH 1.05 -paddingW 1.05 -modelLoc C:\Users\Augus\PycharmProjects\scanner\opencv-text-recognition\frozen_east_text_detection.pb -oem 2 -psm 3

## Test case 4, Image with box, works
# python preProcessing.py -m C:\Users\Augus\PycharmProjects\scanner\images\mjolkny.jpg -t image -show original -morph erosion -morphH 3 -morphW 3 -binarization yes -b1 140 -detection pre-defined -tthresh 2 -dthresh 0.9  -bwl 70 -bwr 15 -bht 0 -bhb 40 -oem 2 -psm 3

## Test case 5, stream with detection, works but very slow
# python preProcessing.py -t video -show original -morph erosion -morphH 3 -morphW 3 -binarization yes -b1 140 -detection detection -tthresh 2 -paddingH 1.05 -paddingW 1.05 -modelLoc C:\Users\Augus\PycharmProjects\scanner\opencv-text-recognition\frozen_east_text_detection.pb -oem 2 -psm 3

## Test case 6, stream with box, works well
# python preProcessing.py -t video -show original -morph erosion -morphH 3 -morphW 3 -binarization yes -b1 140 -detection pre-defined -tthresh 2 -bwl 50 -bwr 50 -bht 20 -bhb 50 -oem 2 -psm 3


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
import time
import pandas as pd
from itertools import chain


def greyscale(coefficients, image):
    coefficients = np.array(coefficients).reshape((1, 3))
    greyImage = cv2.transform(image, coefficients)
    return (greyImage)


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
    return resultImg


def decode_predictions(scores, geometry, args):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < args["dthresh"]:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


def preprocessing(args, image):
    # Greysacle image
    if args["greyscale"] == "yes":
        image = greyscale([args["gR"], args["gG"], args["gB"]], image)

    # Binarization
    if args["binarization"] == "yes":
        ret, thresh1 = cv2.threshold(image, args["bin1"], args["bin2"], cv2.THRESH_BINARY)
        image = cv2.merge((thresh1, thresh1, thresh1))

    # th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    # cv.THRESH_BINARY,11,2)
    # https://docs.opencv.org/3.4.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3

    # Morphology
    if args["morph"] != "none":
        image = morphology(image, iterations=1, flow=args["morph"], sizeW=args["morphW"], sizeH=args["morphH"])

    # Blurring
    if args["blur"] == "yes":
        image = cv2.blur(image, (args["blurH"], args["blurW"]))

    return image


def tesseractDisplay(args, origImage, image, confidences, startW, startH, rW, rH):
    # Config file
    config = ("-l eng --oem " + str(args["oem"]) + " --psm " + str(args["psm"]))

    # Read frame/image
    # print(scores[0][0])
    # print(confidences)
    #print(len(confidences))
    text = ""
    if "detection" == args["detection"] and len(confidences) > args["tthresh"]:
        text = pytesseract.image_to_string(image, config=config)
        cv2.putText(origImage, text, (int(startW * rW), int(startH * rH) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if args["detection"] == "pre-defined":
        text = pytesseract.image_to_string(image, config=config)
        cv2.putText(origImage, text, (int(startW * rW), int(startH * rH) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if args["type"] == "video":
        # Put text on image
        if args["show"] == "original":
            cv2.imshow("Text detection", origImage)
            # key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #     break
        if args["show"] == "edited":
            cv2.imshow("Text detection", image)
            # key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #     break


    if args["type"] == "image":
        if args["show"] == "original":
            (H, W) = origImage.shape[:2]
        smaller = cv2.resize(origImage, (int(round(W / 5)), int(round(H / 5))))
        cv2.imshow("Text detection", smaller)
        cv2.waitKey(0)
        if args["show"] == "edited":
            cv2.imshow("Text detection", image)
        cv2.waitKey(0)

        # display the text OCR'd by Tesseract

    print(text)


## Parser arguments
ap = argparse.ArgumentParser()

## Basic options
# File locations and type
ap.add_argument("-m", "--media", type=str, help="path to input media")
ap.add_argument("-t", "--type", type=str, default="image", help="'video' or 'image'")
ap.add_argument("-show", "--show", type=str, default="edited", help="Show 'original' or 'edited'")

## Pre-processing
# Target image size
ap.add_argument("-imgW", "--targetW", type=int, default=608,
                help="nearest multiple of 32 for resized width, do not change")
ap.add_argument("-imgH", "--targetH", type=int, default=512,
                help="nearest multiple of 32 for resized height, do not change")

# Morphing with options
ap.add_argument("-morph", "--morph", type=str, default="none", help="'erosion', 'dilation', 'opening' or 'closing'?")
ap.add_argument("-morphH", "--morphH", type=int, default=5, help="Morphoplogy, Kernel height")
ap.add_argument("-morphW", "--morphW", type=int, default=5, help="Morphoplogy, Kernel width")

# Blurring with options
ap.add_argument("-blur", "--blur", type=str, default="no", help="blur,'yes' or 'no'?")
ap.add_argument("-blurH", "--blurH", type=int, default=5, help="Blur, Kernel width")
ap.add_argument("-blurW", "--blurW", type=int, default=5, help="Blur, Kernel height")

# Binarization
ap.add_argument("-binarization", "--binarization", type=str, default="yes", help="Apply binarization, 'yes' or 'no'")
ap.add_argument("-b1", "--bin1", type=int, default=140, help="Threshold 1")
ap.add_argument("-b2", "--bin2", type=int, default=255, help="Threshold 2")

# Greyscaling
ap.add_argument("-greyscale", "--greyscale", type=str, default="yes", help="Apply greyscale, 'yes' or 'no'")
ap.add_argument("-gR", "--gR", type=float, default=0.299, help="Greyscale, R")
ap.add_argument("-gG", "--gG", type=float, default=0.587, help="Greyscale, G")
ap.add_argument("-gB", "--gB", type=float, default=0.114, help="Greyscale, B")

# Box selection, when not using detection
ap.add_argument("-bwl", "--boxWL", type=int, default=130,
                help="Box width size, left")
ap.add_argument("-bwr", "--boxWR", type=int, default=130,
                help="Box width size, right")

ap.add_argument("-bht", "--boxHT", type=int, default=35,
                help="Box height size, top")
ap.add_argument("-bhb", "--boxHB", type=int, default=35,
                help="Box height size, bottom")

## Model specific
# EAST model location
ap.add_argument("-modelLoc", "--modelLoc", type=str, default="C:/Path/", help="Model location")

# Object detection or pre-defined box?
ap.add_argument("-detection", "--detection", type=str, default="pre-defined",
                help="Use 'pre-defined' or text 'detection' box")
# Threshold for box
ap.add_argument("-dthresh", "--dthresh", type=float, default=0.9, help="Probability threshold for detection")
ap.add_argument("-tthresh", "--tthresh", type=int, default=2, help="Probability threshold for tesserac activation")

# Padding when using detection
ap.add_argument("-paddingH", "--paddingH", type=float, default=0,
                help="Additional padding for box when using detection, percentage")
ap.add_argument("-paddingW", "--paddingW", type=float, default=0,
                help="Additional padding for box when using detection, percentage")

# Tesserac OCR reader options
ap.add_argument("-oem", "--oem", type=str, default="2", help="OEM SELECTION, model")

# 0 = Original Tesseract only.
# 1 = Neural nets LSTM only.
# 2 = Tesseract + LSTM.
# 3 = Default, based on what is available.

ap.add_argument("-psm", "--psm", type=str, default="3", help="PSM SELECTION, read line")
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

if args["type"] == "video":
    # If video path is not supplied, grab the reference to the web cam
    if not args.get("media", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)

    # Otherwise, Grab a reference to the media file
    else:
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
        orig = frame.copy()

        # Extract original sizes
        (origH, origW) = frame.shape[:2]

        # Proportion of change
        rW = origW / float(args["targetW"])
        rH = origH / float(args["targetH"])

        # Resize image
        frame_resized = cv2.resize(frame, (args["targetW"], args["targetH"]))

        # Extract new sizes
        (H, W) = frame_resized.shape[:2]

        if args["detection"] == "pre-defined":
            # Decide extracted box
            startW, startH = int((W / 2) - args["boxWL"]), int((H / 2) - args["boxHT"])
            endW, endH = int((W / 2) + args["boxWR"]), int((H / 2) + args["boxHB"])

            # extract box
            crop_img = frame_resized[startH:endH, startW:endW]

            # Preprocess image
            preprocessedImage = preprocessing(args, crop_img)

            # Pad image
            crop_img_padded = cv2.copyMakeBorder(preprocessedImage, startH, int(H - endH), startW, int(W - endW),
                                                 borderType,
                                                 None, (255, 255, 255))

            # write box, on original image
            cv2.rectangle(orig, (int(startW * rW), int(startH * rH)), (int(endW * rW), int(endH * rH)), (0, 232, 0), 2)

            tesseractDisplay(args, orig, crop_img_padded, None, startW, startH, rW, rH)

            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if args["detection"] == "detection":

            net = cv2.dnn.readNet(args["modelLoc"])  # Read model

            # define the two output layer names for the EAST detector model that
            # we are interested -- the first is the output probabilities and the
            # second can be used to derive the bounding box coordinates of text
            layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

            # Preprocess image
            preprocessedImage = preprocessing(args, frame_resized)

            # Construct a blob from the frame and then perform a forward pass
            blob = cv2.dnn.blobFromImage(preprocessedImage, 1.0, (W, H),
                                         # (123.68, 116.78, 103.94), # since greyscale
                                         swapRB=True, crop=False)

            net.setInput(blob)
            (scores, geometry) = net.forward(layerNames)

            # decode the predictions, then  apply non-maxima suppression to
            # suppress weak, overlapping bounding boxes
            (rects, confidences) = decode_predictions(scores, geometry, args)
            boxes = non_max_suppression(np.array(rects), probs=confidences)
            print(scores)
            # print("score")
            # print(len(scores))
            # print("conf")
            # print(len(confidences))
            # print("box")
            # print(len(boxes))
            # loop over the bounding boxes
            for (startX, startY, endX, endY) in boxes:
                # Apply padding surrounding the bounding box -- here we
                # are computing the deltas in both the x and y directions
                dX = int((endX - startX) * args["paddingW"])
                dY = int((endY - startY) * args["paddingH"])

                # apply padding to each side of the bounding box, respectively
                startX = max(0, startX - dX)
                startY = max(0, startY - dY)
                endX = min(origW, endX + (dX * 2))
                endY = min(origH, endY + (dY * 2))

                # extract the actual padded ROI
                roi = orig[startY:endY, startX:endX]

                # White-Pad image
                crop_img_padded = cv2.copyMakeBorder(src=roi, top=int((H + (startY - endY)) / 2),
                                                    bottom=int((H - (startY - endY)) / 2),
                                                    left=int((W - (startX - endX)) / 2),
                                                    right=int((W + (startX - endX)) / 2),
                                                    borderType=borderType, value=(255, 255, 255))

                # cv2.imshow("Text detection", crop_img_padded)
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
                #cv2.imshow("Text detection", preprocessedImage)

                # cv2.rectangle(orig, (int((W - (startX - endX)) / 2), int((H - (startY - endY)) / 2)),
                #               (int((W + (startX - endX)) / 2), int((H + (startY - endY)) / 2)), (0, 232, 0), 2)

                tesseractDisplay(args, orig, crop_img_padded, confidences, startX, startY, rW, rH)
                key = cv2.waitKey(1) & 0xFF
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

if args["type"] == "image":
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

    if args["detection"] == "pre-defined":
        # Decide extracted box
        startW, startH = int((W / 2) - args["boxWL"]), int((H / 2) - args["boxHT"])
        endW, endH = int((W / 2) + args["boxWR"]), int((H / 2) + args["boxHB"])

        # extract box
        crop_img = image_resized[startH:endH, startW:endW]

        # Preprocess image
        preprocessedImage = preprocessing(args, crop_img)

        # Pad image
        crop_img_padded = cv2.copyMakeBorder(preprocessedImage, startH, int(H - endH), startW, int(W - endW),
                                             borderType,
                                             None, (255, 255, 255))

        # write box, on original image
        cv2.rectangle(orig, (int(startW * rW), int(startH * rH)), (int(endW * rW), int(endH * rH)), (0, 232, 0), 5)
        # cv2.rectangle(orig, (int(startX * rW), int(startY * rH)), (int(endX * rW), int(endY * rH)), (0, 232, 0), 5)
        tesseractDisplay(args, orig, crop_img_padded, None, startW, startH, rW, rH)

    if args["detection"] == "detection":
        # Read model
        net = cv2.dnn.readNet(args["modelLoc"])

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        # Preprocess image
        preprocessedImage = preprocessing(args, image_resized)

        # Construct a blob from the frame and then perform a forward pass
        blob = cv2.dnn.blobFromImage(preprocessedImage, 1.0, (W, H),
                                     # (123.68, 116.78, 103.94), # since greyscale
                                     swapRB=True, crop=False)

        # Make predictions
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # Decode the predictions, then  apply non-maxima suppression to suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry, args)

        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # initialize the list of results
        results = []

        # Loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # Add padding on boxes
            dX = int((endX - startX) * args["paddingW"])
            dY = int((endY - startY) * args["paddingH"])

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))

            # extract the actual padded ROI
            roi = orig[startY:endY, startX:endX]

            # White-Pad and center image
            crop_img_padded = cv2.copyMakeBorder(src=roi, top=int((H + (startY - endY)) / 2),
                                                 bottom=int((H - (startY - endY)) / 2),
                                                 left=int((W - (startX - endX)) / 2),
                                                 right=int((W + (startX - endX)) / 2),
                                                 borderType=borderType, value=(255, 255, 255))
            cv2.rectangle(orig, (int(startX * rW), int(startY * rH)), (int(endX * rW), int(endY * rH)), (0, 232, 0), 5)
           # cv2.imshow("Text detection", crop_img_padded)
            # cv2.rectangle(orig, (startX*rW, startY*rH), (endX*rW, endY*rH), (0, 255, 0), 2)
            #cv2.waitKey(100000)
            tesseractDisplay(args, orig, crop_img_padded, confidences, startX, startY, rW, rH)
            # def main():
            #     print("Hello World!")
            #
            # if __name__ == "__main__":
            #     main()
