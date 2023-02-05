from os import path

DATASET_TYPE = "age"

BASE_PATH = "./"
OUTPUT_BASE = "./output"
MX_OUTPUT = BASE_PATH

IMAGES_PATH = path.sep.join([BASE_PATH, "aligned"])
LABELS_PATH = path.sep.join([BASE_PATH, "folds"])

NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

BATCH_SIZE = 128

checkpointsPath = path.sep.join([BASE_PATH, "weights"])

if DATASET_TYPE == "age":
    NUM_CLASSES = 8
    LABEL_ENCODER_PATH = path.sep.join([OUTPUT_BASE, "age_le.cpickle"])

    DATASET_MEAN = path.sep.join([OUTPUT_BASE, "age_adience_mean.json"])

elif DATASET_TYPE == "gender":
    NUM_CLASSES = 2
    LABEL_ENCODER_PATH = path.sep.join([OUTPUT_BASE, "gender_le.cpickle"])

    DATASET_MEAN = path.sep.join([OUTPUT_BASE, "gender_adience_mean.json"])


