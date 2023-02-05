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
import torch.utils.data import DataLoader
from torchvision import utils, transforms
import torch.nn.functional as F
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


# TensorBoard可以跟踪损失和准确率、可视化模型图、查看直方图、显示源图像
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints",
                help="path to output checkpoint directory")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = ap.parse_args()

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG,
                    filename="training_{}.log".format(args["start_epoch"]),
                    filemode="w")

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
valloader = DataLoader(dataset=val_data, batch_size=config.BATCH_SIZE / 2, shuffle=False)
testloader = DataLoader(dataset=test_data, batch_size=config.BATCH_SIZE / 2, shuffle=False)

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
optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0005)

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

if args.checkpoints:
    print('Resuming training, loading {}...'.format(args.checkpoints))
    model.load_state_dict(torch.load(args.checkpoints))
    print('Finished!')
else:
    print('Initializing weights...')
    # initialize layers' weights with xavier method
    model.apply(weights_init)

classes = AgeGenderHelper(config).buildAgeBins()

def train_runner(model, device, trainloader, optimizer, epoch):
    # 训练模型, 启用 BatchNormalization 和 Dropout, 将BatchNormalization和Dropout置为True
    model.train()
    total = 0
    correct = 0.0

    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('runs/resnet18'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        # enumerate迭代已加载的数据集,同时获取数据和数据下标
        for i, data in enumerate(trainloader):
            if i >= (1 + 1 + 3) * 2:
                break
            inputs, labels = data
            # 把模型部署到device上
            inputs, labels = inputs.to(device), labels.to(device)
            # create grid of images
            img_grid = utils.make_grid(inputs)
            writer.add_image('images', img_grid)
            writer.add_graph(model, inputs)

            # helper function
            def select_n_random(data, labels):
                '''
                Selects n random datapoints and their corresponding labels from a dataset
                '''
                assert len(data) == len(labels)

                perm = torch.randperm(len(data))
                return data[perm][:len(data)], labels[perm][:len(labels)]

            # select random images and their target indices
            tuxiang, biaoqian = select_n_random(inputs, labels)

            # get the class labels for each image
            class_labels = [classes[lab] for lab in biaoqian]

            # log embeddings
            features = tuxiang.view(3, 1816 * 1816)
            writer.add_embedding(features,
                                # metadata=class_labels,
                                # label_img=inputs.unsqueeze(1),
                                global_step=epoch * len(trainloader) + i
                                 )

            # check to see if the one-off accuracy callback should be used
            if config.DATASET_TYPE == "age":
                # load the label encoder, then build the one-off mappings for
                # computing accuracy
                le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
                agh = AgeGenderHelper(config)
                oneOff = agh.buildOneOffMappings(le)
                acc = one_off_callback(trainloader, valloader, oneOff)
                print("[INFO] one-off: {:.2f}%".format(acc * 100))

            # 保存训练结果
            outputs = model(inputs)
            # 计算损失和
            # 多分类情况通常使用cross_entropy(交叉熵损失函数), 而对于二分类问题, 通常使用sigmod
            loss = F.cross_entropy(outputs, labels).to(device)
            # 初始化梯度
            optimizer.zero_grad()
            # 获取最大概率的预测结果
            # dim=1表示返回每一行的最大值对应的列下标
            predict = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.

            # helper functions
            import matplotlib.pyplot as plt
            import numpy as np
            # show an image(used in the `plot_classes_preds` function below)
            def matplotlib_imshow(img, one_channel=False):
                if one_channel:
                    img = img.mean(dim=0)
                img = img / 2 + 0.5  # unnormalize
                npimg = img.numpy()
                if one_channel:
                    plt.imshow(npimg, cmap="Greys")
                else:
                    plt.imshow(np.transpose(npimg, (1, 2, 0)))

            def images_to_probs(net, images):
                '''
                Generates predictions and corresponding probabilities from a trained
                network and a list of images
                '''
                output = net(images)
                # convert output probabilities to predicted class
                _, preds_tensor = torch.max(output, 1)
                preds = np.squeeze(preds_tensor.numpy())
                return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

            def plot_classes_preds(net, images, labels):
                '''
                Generates matplotlib Figure using a trained network, along with images
                and labels from a batch, that shows the network's top prediction along
                with its probability, alongside the actual label, coloring this
                information based on whether the prediction was correct or not.
                Uses the "images_to_probs" function.
                '''
                preds, probs = images_to_probs(net, images)
                # plot the images in the batch, along with predicted and true labels
                fig = plt.figure(figsize=(12, 48))
                for idx in np.arange(4):
                    ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
                    matplotlib_imshow(images[idx], one_channel=True)
                    ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                        classes[preds[idx]],
                        probs[idx] * 100.0,
                        classes[labels[idx]]),
                        color=("green" if preds[idx] == labels[idx].item() else "red"))
                return fig

            if i % 100 == 0:
                # loss.item()表示当前loss的数值
                print(
                    "Train Epoch{} \t Loss: {:.2f}, accuracy: {:.2f}%".format(epoch, loss.item(), 100 * (correct / total)))
                Loss.append(loss.item())
                Accuracy.append(correct / total)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(model, inputs, labels),
                                  global_step=epoch * len(trainloader) + i)

                print('Saving state, iter:', i)
                torch.save(model.state_dict(), 'weights/ssd300_COCO_' +
                           repr(i) + '.pth')
            torch.save(model.state_dict(), '' + '' + '.pth')

    writer.flush()

    return loss.item(), correct / total


def val_runner(model, device, valloader):
    # 模型验证, 必须要写, 否则只要有输入数据, 即使不训练, 它也会改变权值
    # 因为调用eval()将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
    model.eval()
    # 统计模型正确率, 设置初始值
    correct = 0.0
    val_loss = 0.0
    total = 0

    # 1. gets the probability predictions in a val_size x num_classes Tensor
    # 2. gets the preds in a val_size Tensor
    # takes ~10 seconds to run
    class_probs = []
    class_label = []

    # torch.no_grad将不会计算梯度, 也不会进行反向传播
    with torch.no_grad():
        running_vloss = 0.0
        for data, label in valloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]

            # check to see if the one-off accuracy callback should be used
            if config.DATASET_TYPE == "age":
                # load the label encoder, then build the one-off mappings for
                # computing accuracy
                le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
                agh = AgeGenderHelper(config)
                oneOff = agh.buildOneOffMappings(le)

                # compute and display the one-off evaluation metric
                acc = _compute_one_off(model, valloader, oneOff)
                print("[INFO] one-off: {:.2f}%".format(acc * 100))

            class_probs.append(class_probs_batch)
            class_label.append(label)

            val_loss += F.cross_entropy(output, label).item().to(device)
            running_vloss += val_loss

            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': loss, 'Validation': running_vloss},
                               epoch + 1)
            predict = output.argmax(dim=1)
            # 计算正确数量
            total += label.size(0)
            correct += (predict == label).sum().item()
        # 计算损失值
        print("val_avarage_loss: {:.2f}, accuracy: {:.2f}%".format(val_loss / total, 100 * (correct / total)))

    val_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    val_label = torch.cat(class_label)

    # helper function
    def add_pr_curve_tensorboard(class_index, val_probs, val_label, global_step=0, classes=classes):
        '''
        Takes in a "class_index" from 0 to 9 and plots the corresponding
        precision-recall curve
        '''
        tensorboard_truth = val_label == class_index
        tensorboard_probs = val_probs[:, class_index]

        writer.add_pr_curve(classes[class_index],
                            tensorboard_truth,
                            tensorboard_probs,
                            global_step=global_step)
        writer.flush()

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, val_probs, val_label)

# 调用
epoch = 20
Loss = []
Accuracy = []
for epoch in range(1, epoch + 1):
    print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    loss, acc = train_runner(model, device, trainloader, optimizer, epoch)
    Loss.append(loss)
    Accuracy.append(acc)
    val_runner(model, device, valloader)
    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')

