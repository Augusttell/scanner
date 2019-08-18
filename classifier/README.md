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
