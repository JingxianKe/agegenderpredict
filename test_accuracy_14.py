#! /usr/bin/env python3

# ./test_accuracy.py

# import the necessary packages
from agegenderhelper_14 import AgeGenderHelper
from agegendernet_14 import AgeGenderNet
import age_gender_config_14 as config
from load_data_14 import data_loader
from torch import nn
import pickle
import torch
import os

_, _, test_loader = data_loader(config.NUM_VAL_IMAGES, config.NUM_TEST_IMAGES, config.BATCH_SIZE)

# load the checkpoint from disk
print("[INFO] loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AgeGenderNet(config.NUM_CLASSES).to(device)
checkpointsPath = os.path.sep.join(config.checkpointsPath)
model.load_state_dict(torch.load('./checkpoints/gender/model.pth', map_location=torch.device('cpu')))
model.eval()
print('Finished loading model!')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
test_correct = 0.0
test_test_loss = 0.0
test_total = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_test_loss += criterion(outputs, labels).item()
        predict = outputs.argmax(dim=1)

        # 计算正确数量
        test_total += labels.size(0)

        if config.DATASET_TYPE == "gender":
            test_correct += (predict == labels).sum().item()
        if config.DATASET_TYPE == 'age':
            # load the label encoder, then build the one-off mappings for
            # computing accuracy
            le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
            agh = AgeGenderHelper(config)
            oneOff = agh.buildOneOffMappings(le)

            predict = predict.numpy()
            labels = labels.numpy().astype("int")
            # loop over the predicted labels and ground-truth labels in the batch
            for (pred, label) in zip(predict, labels):
                # if correct label is in the set of "one off"
                # predictions, then update the correct counter
                if label in oneOff[pred]:
                    test_correct += 1

        print(i)

    print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_test_loss / test_total,
                                                                100 * (test_correct / test_total)))
#
# # check to see if the one-off accuracy callback should be used
# if config.DATASET_TYPE == "age":
#     # load the label encoder, then build the one-off mappings for
#     # computing accuracy
#     le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
#     agh = AgeGenderHelper(config)
#     oneOff = agh.buildOneOffMappings(le)
#
#     # compute and display the one-off evaluation metric
#     acc = _compute_one_off(model, test_loader, oneOff)
#     print("[INFO] one-off: {:.2f}%".format(acc * 100))