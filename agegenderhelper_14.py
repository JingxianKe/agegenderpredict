import numpy as np
import glob
import cv2
import os

class AgeGenderHelper:
    def __init__(self, config):
        self.config = config
        self.ageBins = self.buildAgeBins()

    def buildAgeBins(self):
        ageBins = [(0, 2), (4, 6), (8, 13), (15, 20), (25, 32),
                   (38, 43), (48, 53), (60, np.inf)]

        return ageBins

    def toLabel(self, age, gender):
        if self.config.DATASET_TYPE == "age":
            return self.toAgeLabel(age)

        return self.toGenderLabel(gender)

    def toAgeLabel(self, age):
        label = None

        age = age.replace("(", "").replace(")", "").split(", ")
        (ageLower, ageUpper) = np.array(age, dtype="int")

        for (lower, upper) in self.ageBins:
            if ageLower >= lower and ageUpper <= upper:
                label = "{}_{}".format(lower, upper)
                break

        return label

    def toGenderLabel(self, gender):
        return 0 if gender == "m" else 1

    def buildOneOffMappings(self, le):
        classes = sorted(le.classes_, key=lambda x: int(x.split("_")[0]))
        oneOff = {}

        for (i, name) in enumerate(classes):
            current = np.where(le.classes_ == name)[0][0]
            prev = -1
            next = -1

            if i > 0:
                prev = np.where(le.classes_ == classes[i - 1])[0][0]

            if i < len(classes) - 1:
                next = np.where(le.classes_ == classes[i + 1])[0][0]

            oneOff[current] = (current, prev, next)

        return oneOff

    def buildPathsAndLabels(self):
        paths = []
        labels = []

        foldPaths = os.path.sep.join([self.config.LABELS_PATH, "*.txt"])
        foldPaths = glob.glob(foldPaths)

        for foldPath in foldPaths:
            rows = open(foldPath).read()
            rows = rows.strip().split("\n")[1:]

            for row in rows:
                row = row.split("\t")
                (userID, imagePath, faceID, age, gender) = row[:5]

                if age[0] != "(" or gender not in ("m", "f"):
                    continue

                p = "landmark_aligned_face.{}.{}".format(faceID, imagePath)
                p = os.path.sep.join([self.config.IMAGES_PATH, userID, p])
                label = self.toLabel(age, gender)

                if label is None:
                    continue

                paths.append(p)
                labels.append(label)

        return (paths, labels)

    @staticmethod
    def visualizeAge(agePreds, le):
        canvas = np.zeros((250, 310, 3), dtype="uint8")
        idxs = np.argsort(agePreds)[::-1]

        for (i, j) in enumerate(idxs):
            ageLabel = le.inverse_transform(j).decode("utf-8")
            ageLabel = ageLabel.replace("_", "-")
            ageLabel = ageLabel.replace("-inf", "+")
            text = "{}: {:.2f}%".format(ageLabel, agePreds[j] * 100)

            w = int(agePreds[j] * 300) + 5
            cv2.rectangle(canvas, (5, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        return canvas

    @staticmethod
    def visualizeGender(genderPreds, le):
        canvas = np.zeros((100, 310, 3), dtype="uint8")
        genderPreds = genderPreds.numpy()
        idxs = np.argsort(genderPreds)[::-1]

        for (i, j) in enumerate(idxs):
            gender = "Male" if j == 0 else "Female"
            text = "{}: {:.2f}%".format(gender, genderPreds[j] * 100)

            w = int(genderPreds[j] * 300) + 5
            cv2.rectangle(canvas, (5, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        return canvas

