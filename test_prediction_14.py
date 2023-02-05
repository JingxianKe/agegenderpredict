#! /usr/bin/env python3

# import the necessary packages
import age_gender_config_14 as config
import age_gender_deploy_14 as deploy

from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from croppreprocessor import CropPreprocessor
from agegenderhelper_14 import AgeGenderHelper
from agegendernet_14 import AgeGenderNet
from facealigner_14 import FaceAligner
from torchvision import transforms
from imutils import face_utils
from imutils import paths
from PIL import Image
import numpy as np
import argparse
import pickle
import imutils
import torch
import json
import dlib
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image (or directory)")
args = vars(ap.parse_args())

# load the label encoders and mean files
print("[INFO] loading label encoders and mean files...")
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())
genderMeans = json.loads(open(deploy.GENDER_MEANS).read())

# load the models from disk
print("[INFO] loading models...")
agePath = os.path.sep.join([deploy.AGE_NETWORK_PATH,
                            deploy.AGE_PREFIX])
genderPath = os.path.sep.join([deploy.GENDER_NETWORK_PATH,
                               deploy.GENDER_PREFIX])


# load the checkpoint from disk
print("[INFO] loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
genderModel = AgeGenderNet(config.NUM_CLASSES).to(device)
checkpointsPath = os.path.sep.join(config.checkpointsPath)
genderModel.load_state_dict(torch.load('./checkpoints/gender/model.pth', map_location=torch.device('cpu')))
genderModel.eval()
print('Finished loading model!')

# initialize the image pre-processors
sp = SimplePreprocessor(width=256, height=256,
                        inter=cv2.INTER_CUBIC)
cp = CropPreprocessor(width=227, height=227, horiz=True)
genderMP = MeanPreprocessor(genderMeans["R"], genderMeans["G"],
                            genderMeans["B"])
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
preprocess_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    normalize,
])

# initialize dlib’s face detector (HOG-based), then create the
# the facial landmark predictor and face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

# initialize the list of image paths as just a single image
imagePaths = [args["image"]]

# if the input path is actually a directory, then list all image
# paths in the directory
if os.path.isdir(args["image"]):
    imagePaths = sorted(list(paths.list_files(args["image"])))

# loop over the image paths
for imagePath in imagePaths:
    # load the image from disk, resize it, and convert it to
    # grayscale
    print("[INFO] processing {}".format(imagePath))
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # align the face
        shape = predictor(gray, rect)
        face = fa.align(image, gray, rect)

        # resize the face to a fixed size, then extract 10-crop
        # patches from it
        face = sp.preprocess(face)
        patches = cp.preprocess(face)

        # draw the bounding box around the face
        clone = image.copy()
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the output image
        cv2.imshow("Input", clone)
        cv2.imshow("Face", face)
        cv2.waitKey(0)





