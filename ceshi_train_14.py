#       项目结构：
#       年龄性别预测
#       |---aligned
#       |   |---7153718@N04 ...
#       |   |   |---.jpg ...
#       |---checkpoints
#       |   |---age
#       |   |---gender
#       |---folds
#       |   |---fold_frontal_0_data.txt
#       |   |---fold_frontal_..._data.txt
#       |   |---fold_frontal_4_data.txt
#       |---output
#       |   |---age_adience_mean.json
#       |   |---age_le.cpickle
#       |   |---gender_adience_mean.json
#       |   |---gender_le.cpickle
#       |---runs
#       |   |---
#       |   |---
#       |---weights
#       |   |---
#       |---age_gender_config_14.py
#       |---age_gender_deploy_14.py
#       |---agegenderhelper_14.py
#       |---agegendernet_14.py
#       |---build_dataset_14.py
#       |---metrics_14.py
#       |---tensorboardclass_14.py
#       |---train_14.py
#       |---preprocessor_14.py
#       |---vis_classification_14.py


import age_gender_config_14 as config
from build_dataset_14 import MyDataset
from agegendernet_14 import AgeGenderNet
from metrics_14 import one_off_callback
from agegenderhelper_14 import AgeGenderHelper
import argparse
import logging
import time
import json
import pickle

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import utils, transforms
import torch.nn.functional as F
# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
#
#
# # TensorBoard可以跟踪损失和准确率、可视化模型图、查看直方图、显示源图像
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
#
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--checkpoints",
#                 help="path to output checkpoint directory")
# ap.add_argument("-s", "--start-epoch", type=int, default=0,
#                 help="epoch to restart training at")
# args = ap.parse_args()

# # set the logging level and output file
# logging.basicConfig(level=logging.DEBUG,
#                     filename="training_{}.log".format(args["start_epoch"]),
#                     filemode="w")

means = json.loads(open(config.DATASET_MEAN).read())

pipline_train = transforms.Compose([
    #随机旋转图片
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=7),
    transforms.RandomCrop((227, 227)),
    #将图片转化为Tensor格式
    transforms.ToTensor(),
    #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
    transforms.Normalize((means["R"] / 255.0, means["G"] / 255.0, means["B"] / 255.0), (1.0, 1.0, 1.0))
    #transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])
pipline_val_and_test = pipline_test = transforms.Compose([
    #将图片尺寸resize到227x227
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize((means["R"] / 255.0, means["G"] / 255.0, means["B"] / 255.0), (1.0, 1.0, 1.0))
    #transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])

train_data = MyDataset(transform=pipline_train, state='train')
val_data = MyDataset(transform=pipline_val_and_test, state='val')
test_data = MyDataset(transform=pipline_val_and_test, state='test')

trainloader = DataLoader(dataset=train_data, batch_size=config.BATCH_SIZE, shuffle=True)
valloader = DataLoader(dataset=val_data, batch_size=int(config.BATCH_SIZE / 2), shuffle=True)
testloader = DataLoader(dataset=test_data, batch_size=int(config.BATCH_SIZE / 2), shuffle=True)

# examples = enumerate(trainloader)
# batch_idx, (example_data, example_label) = next(examples)

# # 批量展示图片
# import matplotlib.pyplot as plt
# import numpy as np
# for i in range(4):
#     plt.subplot(1, 4, i + 1)
#     plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
#     img = train_data[i][0]
#     img = img.numpy()  # FloatTensor转为ndarray
#     img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
#     img = img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]
#     plt.imshow(img)
#     plt.title("label:{}".format(example_label[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


# 创建模型，部署gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeGenderNet(config.NUM_CLASSES).to(device)

# initialize the optimizer
optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)

# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)
#
# if args.checkpoints:
#     print('Resuming training, loading {}...'.format(args.checkpoints))
#     model.load_state_dict(torch.load(args.checkpoints))
#     print('Finished!')
# else:
#     print('Initializing weights...')
#     # initialize layers' weights with xavier method
#     model.apply(weights_init)

# classes = AgeGenderHelper(config).buildAgeBins()
#
# def train_runner(model, device, trainloader, optimizer):
#     # 训练模型, 启用 BatchNormalization 和 Dropout, 将BatchNormalization和Dropout置为True
#     # model.train()
#     running_loss = 0
#     last_loss = 0
#         # enumerate迭代已加载的数据集,同时获取数据和数据下标
#     for i, data in enumerate(trainloader):
#         inputs, labels = data
#         # 把模型部署到device上
#         inputs, labels = inputs.to(device), labels.to(device)
#         # 初始化梯度
#         optimizer.zero_grad()
#
#         # Make predictions for this batch
#         outputs = model(inputs)
#
#         gender_criterion = nn.CrossEntropyLoss()
#         loss = gender_criterion(outputs, labels).to(device)
#
#         loss.backward()
#
#         # Adjust learning weights
#         optimizer.step()
#
#         # Gather data and report
#         running_loss += loss.item()
#         if i % 10 == 9:
#             last_loss = running_loss / 10  # loss per batch
#             print('  batch {} loss: {}'.format(i + 1, last_loss))
#             running_loss = 0
#
#     return last_loss
#
#
# def val_runner(model, device, valloader):
#     #模型验证, 必须要写, 否则只要有输入数据, 即使不训练, 它也会改变权值
#     #因为调用eval()将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
#     model.eval()
#     #统计模型正确率, 设置初始值
#     correct = 0.0
#     test_loss = 0.0
#     total = 0
#     #torch.no_grad将不会计算梯度, 也不会进行反向传播
#     with torch.no_grad():
#         for data, label in valloader:
#             data, label = data.to(device), label.to(device)
#             output = model(data)
#             gender_criterion = nn.CrossEntropyLoss()
#             test_loss += gender_criterion(output, label).item()
#             predict = output.argmax(dim=1)
#             #计算正确数量
#             total += label.size(0)
#             correct += (predict == label).sum().item()
#         #计算损失值
#         print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss/total, 100*(correct/total)))
#
# # 调用
# epoch = 120
# Loss = []
# Accuracy = []
# for epoch in range(1, epoch + 1):
#     print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
#     loss, acc = train_runner(model, device, trainloader, optimizer)
#     Loss.append(loss)
#     Accuracy.append(acc)
#     val_runner(model, device, valloader)
#     print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')



num_classes = 10
num_epochs = 20
criterion = nn.CrossEntropyLoss()
total_step = len(trainloader)

for epoch in range(num_epochs):
    print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for i, (images, labels) in enumerate(trainloader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')