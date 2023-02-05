#! /usr/bin/env python3

# import the necessary packages
import time
import torch
import pickle
import logging
import argparse
import torch.nn as nn
from torchvision import utils
from load_data_14 import data_loader
from metrics_14 import one_off_callback
import age_gender_config_14 as config
from agegendernet_14 import AgeGenderNet
from agegenderhelper_14 import AgeGenderHelper

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

# TensorBoard可以跟踪损失和准确率、可视化模型图、查看直方图、显示源图像
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG,
                    filename="training_{}.log".format(args["start_epoch"]),
                    filemode="w")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
train_loader, valid_loader, _ = data_loader(config.NUM_VAL_IMAGES, config.NUM_TEST_IMAGES, config.BATCH_SIZE)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

num_epochs = 80
batch_size = 128
learning_rate = 0.01

print("[INFO] building network...")
model = AgeGenderNet(config.NUM_CLASSES).to(device)

if args["start_epoch"] > 0:
    # load the checkpoint from disk
    print("[INFO] loading epoch {}...".format(args["start_epoch"]))
    model.load_state_dict(torch.load(config.checkpointsPath))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

# # check to see if the one-off accuracy callback should be used
# if config.DATASET_TYPE == "age":
#     # load the label encoder, then build the one-off mappings for
#     # computing accuracy
#     le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
#     agh = AgeGenderHelper(config)
#     oneOff = agh.buildOneOffMappings(le)
#     epochEndCBs.append(one_off_callback(model, train_loader, valid_loader, oneOff))

# Train the model
total_step = len(train_loader)

print('training!')
since = time.time()
for epoch in range(1, num_epochs):
    model.train()
    total = 0
    test_loss = 0.0
    correct = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # create grid of images and write to tensorboard
        img_grid = utils.make_grid(images)
        writer.add_image('images', img_grid)
        writer.add_graph(model, images)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        predict = outputs.argmax(dim=1)
        total += labels.size(0)

        if config.DATASET_TYPE == "gender":
            correct += (predict == labels).sum().item()
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
                    correct += 1

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('reusing', i)
    if epoch % 5 == 0:
        print(f"---- Saving checkpoint to: './model.pth' ----")
        torch.save(model.state_dict(), './model.pth')

    print("Train Epoch{} \t Loss: {:.6f}, accuracy: {:.6f}%".format(epoch, test_loss / total, 100 * (correct / total)))

    # Validation
    model.eval()
    val_correct = 0.0
    val_test_loss = 0.0
    val_total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_test_loss += criterion(outputs, labels).item()
            predict = outputs.argmax(dim=1)
            # 计算正确数量
            val_total += labels.size(0)
            val_correct += (predict == labels).sum().item()

        print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(val_test_loss / val_total, 100 * (val_correct / val_total)))

    writer.add_scalars('Train/Loss', {
        'Train': test_loss / total,
        'Validate': val_test_loss / val_total},
                       epoch)
    writer.add_scalars('Train/Accuracy', {
        'Train': 100 * (correct / total),
        'Validate': 100 * (val_correct / val_total)},
                       epoch)

    # Usage: tensorboard --logdir=runs

    writer.flush()
time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')