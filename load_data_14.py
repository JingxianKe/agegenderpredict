import torch
from torch.utils.data import random_split
from build_dataset_14 import MyDataset, SplitDataset

def data_loader(NUM_IMAGES, num_images, batch_size):
    val_size = int(len(MyDataset()) * NUM_IMAGES)
    test_size = int(len(MyDataset()) * num_images)
    train_size = len(MyDataset()) - val_size - test_size

    # load the dataset
    train_dataset, valid_dataset, test_dataset = random_split(MyDataset(),
                                                              [train_size, val_size, test_size])

    train_dataset, test_dataset, valid_dataset = SplitDataset(train_dataset, 'train'), SplitDataset(test_dataset, 'test'), SplitDataset(valid_dataset, 'val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
