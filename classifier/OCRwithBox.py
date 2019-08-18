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


def decode_predictions(scores, geometry):
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
            if scoresData[x] < args["min_confidence"]:
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
ap.add_argument("-imgW", "--targetW", type=int, default=600,
                help="nearest multiple of 32 for resized width, do not change")
ap.add_argument("-imgH", "--targetH", type=int, default=500,
                help="nearest multiple of 32 for resized height, do not change")

# Binarization
ap.add_argument("-binarization", "--binarization", type=str, default="yes", help="Apply binarization, 'yes' or 'no'")
ap.add_argument("-b1", "--bin1", type=int, default=140, help="Threshold 1")
ap.add_argument("-b2", "--bin2", type=int, default=255, help="Threshold 2")

# Greyscaling
ap.add_argument("-greyscale", "--greyscale", type=str, default="yes", help="Apply greyscale, 'yes' or 'no'")
ap.add_argument("-gR", "--gR", type=float, default=0.299, help="Greyscale, R")
ap.add_argument("-gG", "--gG", type=float, default=0.587, help="Greyscale, G")
ap.add_argument("-gB", "--gB", type=float, default=0.114, help="Greyscale, B")

# Object detection or pre-defined box?
ap.add_argument("-detection", "--detection", type=str, default="pre-defined",
                help="Use 'pre-defined' or text 'detection' box")
ap.add_argument("-dthresh", "--dthresh", type=float, default="pre-defined", help="Probability threshold for detection")

# Box selection
ap.add_argument("-bwl", "--boxWL", type=int, default=130,
                help="Box width size, left")
ap.add_argument("-bwr", "--boxWR", type=int, default=130,
                help="Box width size, right")

ap.add_argument("-bht", "--boxHT", type=int, default=35,
                help="Box height size, top")
ap.add_argument("-bhb", "--boxHB", type=int, default=35,
                help="Box height size, bottom")


# Padding
ap.add_argument("-paddingH", "--paddingH", type=float, default=0,
                help="Additional padding for box when using detection")
ap.add_argument("-paddingW", "--paddingW", type=float, default=0,
                help="Additional padding for box when using detection")



# Tesserac OCR reader options
ap.add_argument("-modelLoc", "--modelLoc", type=str, default="C:/Path/", help="Model location")

# Tesserac OCR reader options
ap.add_argument("-oem", "--oem", type=str, default="1", help="OEM SELECTION, model")
# 0 = Original Tesseract only.
# 1 = Neural nets LSTM only.
# 2 = Tesseract + LSTM.
# 3 = Default, based on what is available.

ap.add_argument("-psm", "--psm", type=str, default="1", help="PSM SELECTION, read line")
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

# TODO add detection for video
# TODO add detection for image
# TODO add streaming compatability
# TODO test
# TODO add variable for greyscale of video detection algo?
# TODO Add conditional for if video detect pict, send to tesserac this is going to be main thing later
# TODO Clean up code afterwards, getting long need more functions/methods
# Figure out if one should  extract a priori a little box in the center? - help as much as possible.

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

            # Greysacle image
            if args["greyscale"] == "yes":
                crop_img = greyscale([args["gR"], args["gG"], args["gB"]], crop_img)

            # Binarization
            if args["binarization"] == "yes":
                ret, thresh1 = cv2.threshold(crop_img, args["bin1"], args["bin2"], cv2.THRESH_BINARY)
                crop_img = cv2.merge((thresh1, thresh1, thresh1))

            # Morphology
            if args["morph"] != "none":
                crop_img = morphology(crop_img, iterations=1, flow=args["morph"], sizeW=args["morphW"],
                                      sizeH=args["morphH"])

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

            # Put text on image
            cv2.putText(orig, text, (int(startW * rW), int(startH * rH) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            if args["show"] == "original":
                cv2.imshow("Text detection", orig)
            if args["show"] == "edited":
                cv2.imshow("Text detection", crop_img_padded)

            key = cv2.waitKey(1) & 0xFF

            # display the text OCR'd by Tesseract
            # print("OCR TEXT")
            # print("========")
            print(text)

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        if args["detection"] == "detection":

            net = cv2.dnn.readNet(args["modelLoc"])  # Read model

            # define the two output layer names for the EAST detector model that
            # we are interested -- the first is the output probabilities and the
            # second can be used to derive the bounding box coordinates of text
            layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

            # TODO insert pre-processing here?

            # Greysacle image
            if args["greyscale"] == "yes":
                frame_resized = greyscale([args["gR"], args["gG"], args["gB"]], frame_resized)

            # Binarization
            if args["binarization"] == "yes":
                ret, thresh1 = cv2.threshold(frame_resized, args["bin1"], args["bin2"], cv2.THRESH_BINARY)
                frame_resized = cv2.merge((thresh1, thresh1, thresh1))

            # Morphology
            if args["morph"] != "none":
                frame_resized = morphology(frame_resized, iterations=1, flow=args["morph"], sizeW=args["morphW"],
                                      sizeH=args["morphH"])

            # Blurring
            if args["blur"] == "yes":
                frame_resized = cv2.blur(frame_resized, (args["blurH"], args["blurW"]))

            # of the model to obtain the two output layer sets
            # construct a blob from the frame and then perform a forward pass
            blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (W, H),
                                         #(123.68, 116.78, 103.94), # since greyscale
                                         swapRB=True, crop=False)
            # Parameters
            #     images	input images (all with 1-, 3- or 4-channels).
            #     size	spatial size for output image
            #     mean	scalar with mean values which are subtracted from channels. Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR ordering and swapRB is true.
            #     scalefactor	multiplier for images values.
            #     swapRB	flag which indicates that swap first and last channels in 3-channel image is necessary.
            #     crop	flag which indicates whether image will be cropped after resize or not
            #     ddepth	Depth of output blob. Choose CV_32F or CV_8U.

            net.setInput(blob)
            (scores, geometry) = net.forward(layerNames)

            # decode the predictions, then  apply non-maxima suppression to
            # suppress weak, overlapping bounding boxes
            (rects, confidences) = decode_predictions(scores, geometry)
            boxes = non_max_suppression(np.array(rects), probs=confidences)

            # initialize the list of results
            results = []

            # loop over the bounding boxes
            for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
                # startX = int(startX * rW)
                # startY = int(startY * rH)
                # endX = int(endX * rW)
                # endY = int(endY * rH)

                # in order to obtain a better OCR of the text we can potentially
                # apply a bit of padding surrounding the bounding box -- here we
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
                # crop_img_padded = cv2.copyMakeBorder(roi, startH, int(H - endH), startW, int(W - endW), borderType, # These should be as large as original image
                #                                     None, (255, 255, 255))
                crop_img_padded = cv2.copyMakeBorder(src=roi, top=int((H+(startY-endY)) / 2),
                                                     bottom=int((H-(startY-endY)) / 2),
                                                     left=int(((W-(startX-endX)) / 2),
                                                              right=int((W+(startX-endX)) / 2),
                                                              borderType, None, (255, 255, 255))

                # Config for OCR reader
                # config = ("-l eng --oem " + args["oem"] + " --psm " + args["psm"] + " -c tessedit_char_whitelist=0123456789")
                config = ("-l eng --oem " + str(args["oem"]) + " --psm " + str(args["psm"]))

                # text = pytesseract.image_to_string(roi, config=config)
                text = pytesseract.image_to_string(crop_img_padded, config=config)

                # Put text on image
                cv2.putText(orig, text, (int(startW * rW), int(startH * rH) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                if args["show"] == "original":
                    cv2.imshow("Text detection", orig)
                if args["show"] == "edited":
                    cv2.imshow("Text detection", crop_img_padded)

                key = cv2.waitKey(1) & 0xFF

                # display the text OCR'd by Tesseract
                # print("OCR TEXT")
                # print("========")
                print(text)

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break




            # loop over the bounding boxes
            for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                # draw the bounding box on the frame
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)



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

if args["detection"] == "pre-defined":
    # Decide extracted box
    startW, startH = int((W / 2) - args["boxWL"]), int((H / 2) - args["boxHT"])
    endW, endH = int((W / 2) + args["boxWR"]), int((H / 2) + args["boxHB"])

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
    crop_img_padded = cv2.copyMakeBorder(crop_img, startH, int(H - endH), startW, int(W - endW), borderType, None,
                                         (255, 255, 255))

    # write box, on original image
    cv2.rectangle(image, (int(startW * rW), int(startH * rH)), (int(endW * rW), int(endH * rH)), (0, 232, 0), 5)

if args["detection"] == "detection":

# Config for OCR reader
# config = ("-l eng --oem " + args["oem"] + " --psm " + args["psm"] + " -c tessedit_char_whitelist=0123456789")
config = ("-l eng --oem " + str(args["oem"]) + " --psm " + str(args["psm"]))

# text = pytesseract.image_to_string(roi, config=config)
text = pytesseract.image_to_string(crop_img_padded, config=config)
# testText = "test"

# Put text on image
cv2.putText(image, text, (int(startW * rW), int(startH * rH) - 20),
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

