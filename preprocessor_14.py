
import numpy as np
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)

class ImageToArrayPreprocessor:
    def __init__(self):
        pass

    def preprocess(self, image):
        return np.asarray(image, dtype='float32')

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        # store the Red, Green, and Blue channel averages across a
        # training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        # split the image into its respective Red, Green, and Blue
        # channels
        (B, G, R) = cv2.split(image.astype("float32"))

        # subtract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # merge the channels back together and return the image
        return cv2.merge([B, G, R])

class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        # store the target image width, height, whether or not
        # horizontal flips should be included, along with the
        # interpolation method used when resizing
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image):
        # initialize the list of crops
        crops = []

        # grab the width and height of the image then use these
        # dimensions to define the corners of the image based
        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]]

        # compute the center crop of the image as well
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        # loop over the coordinates, extract each of the crops,
        # and resize each of them to a fixed size
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height),
                              interpolation=self.inter)
            crops.append(crop)

        # check to see if the horizontal flips should be taken
        if self.horiz:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        # return the set of crops
        return np.array(crops)

