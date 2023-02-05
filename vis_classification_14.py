#! /usr/bin/env python3

# import the necessary packages
import age_gender_config_14 as config
import age_gender_deploy_14 as deploy
from agegendernet_14 import AgeGenderNet
from agegenderhelper_14 import AgeGenderHelper
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import pickle
import imutils
import torch
import json
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sample-size", type=int, default=10,
                help="epoch # to load")
args = ap.parse_args()

# load the label encoders and mean files
print("[INFO] loading label encoders and mean files...")
ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER, "rb").read())
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())
genderMeans = json.loads(open(deploy.GENDER_MEANS).read())

# load the checkpoint from disk
print("[INFO] loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
genderModel = AgeGenderNet(config.NUM_CLASSES).to(device)
checkpointsPath = os.path.sep.join(config.checkpointsPath)
genderModel.load_state_dict(torch.load('./checkpoints/gender/model.pth', map_location=torch.device('cpu')))
genderModel.eval()
print('Finished loading model!')

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
preprocess_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    normalize,
])

agh = AgeGenderHelper(config)
(Paths, Labels) = agh.buildPathsAndLabels()


with torch.no_grad():
    # loop over the rows
    for i in range(7):
        imagePath = Paths[i]
        gtLabel = Labels[i]
        image = cv2.imread(imagePath)
        genderImage = Image.open(imagePath)
        genderImage = preprocess_transform(genderImage)
        genderImage.unsqueeze_(0)
        genderImage = genderImage.to(device)
        outputs = genderModel(genderImage)
        genderPreds = outputs.argmax(dim=1)
        genderCanvas = agh.visualizeGender(genderPreds, genderLE)

        image = imutils.resize(image, width=400)
        # show the output image
        cv2.imshow("Image", image)
        cv2.imshow("Gender Probabilities", genderCanvas)
        cv2.waitKey(0)


