# import the necessary packages
import age_gender_config_14 as config
from agegenderhelper_14 import AgeGenderHelper
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle

# initialize our helper class, then build the set of image paths and class labels
# print("[INFO] building paths and labels...")
agh = AgeGenderHelper(config)
(Paths, Labels) = agh.buildPathsAndLabels()

# our class labels are represented as strings so we need to encode them
# print("[INFO] encoding labels...")
le = LabelEncoder().fit(Labels)
Labels = le.transform(Labels)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

train_transform = transforms.Compose([
    # 随机旋转图片
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=7),
    transforms.RandomCrop((227, 227)),
    # 将图片转化为Tensor格式
    transforms.ToTensor(),
    # 正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
    normalize,
    ])

val_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    normalize,
])

class MyDataset(Dataset):
    def __init__(self):
        self.imgs = Paths
        self.label = Labels

    def __getitem__(self, index):
        fn = self.imgs[index]
        label = self.label[index]
        img = Image.open(fn).convert('RGB')

        return img, label

    def __len__(self):
        return len(self.imgs)

class SplitDataset:
    def __init__(self, dataset, split):
        self.ds = dataset
        self.split = split

    def __getitem__(self, idx):
        img = self.ds[idx][0]
        label = self.ds[idx][1]
        if self.split == 'train':
            img = train_transform(img)
        elif self.split == 'test':
            img = test_transform(img)
        else:
            img = val_transform(img)
        return img, label

    def __len__(self):
        return len(self.ds)

####################################################################################
    # label_df = pd.read_csv('/content/labels.csv')
    # print('Training set: {}'.format(label_df.shape))
    #           output: raining set: (10222, 2)

    # train_df, test_df = train_test_split(label_df, test_size=0.1, random_state=0)
    #
    # train_df.shape, test_df.shape
    #
    # # Create dataloaders form datasets
    # train_set = DogDataset(train_df, transform=train_transformer)
    # val_set = DogDataset(test_df, transform=val_transformer)
    #
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    #
    # dataset_sizes = len(train_set)
    #
    # print(dataset_sizes, len(val_set))
####################################################################################
    # # Get a batch of training data
    # inputs, classes = next(iter(val_loader))
    # classes = classes.numpy()
####################################################################################
    # figure = plt.figure(figsize=(12, 12))
    # cols, rows = 4, 4
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(val_set), size=(1,)).item()
    #
    #     inp = inputs[i - 1].numpy().transpose((1, 2, 0))
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     inp = std * inp + mean
    #     inp = np.clip(inp, 0, 1)
    #
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(index_to_breed[classes[i - 1]])
    #     plt.axis("off")
    #     plt.imshow(inp)
    # plt.show()
####################################################################################



#
# def rgb_mean(img):
#     (R, G, B) = ([], [], [])
#
#     _, (r, g, b) = img.getcolors()
#     R.append(r)
#     G.append(g)
#     B.append(b)
#
#     # construct a dictionary of averages, then serialize the means to a
#     # JSON file
#     print("[INFO] serializing means...")
#     D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
#     f = open(config.DATASET_MEAN, "w")
#     f.write(json.dumps(D))
#     f.close()
#
#     return f
#
# def serialize_label():
    # write the label encoder to file
    # print("[INFO] serializing label encoder...")
    f = open(config.LABEL_ENCODER_PATH, "wb")
    f.write(pickle.dumps(le))
    f.close()
    #
    # return f