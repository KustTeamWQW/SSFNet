import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def addZeroPadding(X, margin=2):
    """
    add zero padding to the image
    """
    newX = np.zeros((
        X.shape[0] + 2 * margin,
        X.shape[1] + 2 * margin,
        X.shape[2]
    ))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX

def createImgCube(X, gt, pos: list, windowSize=25):
    """
    create Cube from pos list
    return imagecube gt nextPos
    """

    margin = (windowSize - 1) // 2
    zeroPaddingX = addZeroPadding(X, margin=margin)
    dataPatches = np.zeros((pos.__len__(), windowSize, windowSize, X.shape[2]))
    if (pos[-1][1] + 1 != X.shape[1]):
        nextPos = (pos[-1][0], pos[-1][1] + 1)
    elif (pos[-1][0] + 1 != X.shape[0]):
        nextPos = (pos[-1][0] + 1, 0)
    else:
        nextPos = (0, 0)
    return np.array([zeroPaddingX[i:i + windowSize, j:j + windowSize, :] for i, j in pos]), \
        np.array([gt[i, j] for i, j in pos]), \
        nextPos


def createPos(shape: tuple, pos: tuple, num: int):
    """
    creatre pos list after the given pos
    """
    if (pos[0] + 1) * (pos[1] + 1) + num > shape[0] * shape[1]:
        num = shape[0] * shape[1] - ((pos[0]) * shape[1] + pos[1])
    return [(pos[0] + (pos[1] + i) // shape[1], (pos[1] + i) % shape[1]) for i in range(num)]


def createPosWithoutZero(hsi, gt):
    """
    creatre pos list without zero labels
    """
    mask = gt > 0
    return [(i, j) for i, row in enumerate(mask) for j, row_element in enumerate(row) if row_element]


def splitTrainTestSet(X, gt, valRatio, randomState=111):
    """
    random split data set
    """
    X_train, X_val, gt_train, gt_val = train_test_split(X, gt, test_size=valRatio, random_state=randomState,
                                                          stratify=gt)
    return X_train, X_val, gt_train, gt_val


def createImgPatch(lidar, pos: list, windowSize=25):
    """
    return lidar Img patches
    """
    margin = (windowSize - 1) // 2
    zeroPaddingLidar = np.zeros((
        lidar.shape[0] + 2 * margin,
        lidar.shape[1] + 2 * margin
    ))
    zeroPaddingLidar[margin:lidar.shape[0] + margin, margin:lidar.shape[1] + margin] = lidar
    return np.array([zeroPaddingLidar[i:i + windowSize, j:j + windowSize] for i, j in pos])


def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def data_aug(train_hsiCube, train_patches, train_labels):
    Xh = []
    Xl = []
    y = []
    for i in range(train_hsiCube.shape[0]):
        Xh.append(train_hsiCube[i])
        Xl.append(train_patches[i])

        noise = np.random.normal(0.0, 0.02, size=train_hsiCube[0].shape)
        noise2 = np.random.normal(0.0, 0.02, size=train_patches[0].shape)
        Xh.append(np.flip(train_hsiCube[i] + noise, axis=1))
        Xl.append(np.flip(train_patches[i] + noise2, axis=1))

        k = np.random.randint(4)
        Xh.append(np.rot90(train_hsiCube[i], k=k))
        Xl.append(np.rot90(train_patches[i], k=k))

        y.append(train_labels[i])
        y.append(train_labels[i])
        y.append(train_labels[i])

    train_labels = np.asarray(y, dtype=np.int8)
    train_hsiCube = np.asarray(Xh, dtype=np.float32)
    train_patches = np.asarray(Xl, dtype=np.float32)
    train_hsiCube = torch.from_numpy(train_hsiCube.transpose(0, 3, 1, 2)).float()
    train_patches = torch.from_numpy(train_patches.transpose(0, 3, 1, 2)).float()
    return train_hsiCube, train_patches, train_labels


# class TensorDataset(torch.utils.data.Dataset):
#     def __init__(self, hsi, msi, labels, include_msi=True):
#         self.len = labels.shape[0]
#         self.include_msi = include_msi
#         self.hsi = torch.FloatTensor(hsi)
#         if include_msi:
#             self.msi = torch.FloatTensor(msi)
#         self.labels = torch.LongTensor(labels - 1)
#
#     def __getitem__(self, index):
#         if self.include_msi:
#             return self.hsi[index], self.msi[index], self.labels[index]
#         else:
#             return self.hsi[index], self.labels[index]
#
#     def __len__(self):
#         return self.len

class TensorDataset1(torch.utils.data.Dataset):
    def __init__(self, hsi, msi, labels):
        self.len = labels.shape[0]
        self.hsi = torch.FloatTensor(hsi)
        self.msi = torch.FloatTensor(msi)
        self.labels = torch.LongTensor(labels - 1)
    def __getitem__(self, index):
        return self.hsi[index], self.msi[index], self.labels[index]
    def __len__(self):
        return self.len


class TensorDataset2(torch.utils.data.Dataset):
    def __init__(self, hsi, labels):
        self.len = labels.shape[0]
        self.hsi = torch.FloatTensor(hsi)
        self.labels = torch.LongTensor(labels - 1)
    def __getitem__(self, index):
        return self.hsi[index],  self.labels[index]
    def __len__(self):
        return self.len


import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader
from collections import Counter

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader
from collections import Counter


def build_datasets(root, dataset, patch_size, batch_size, val_ratio, include_msi=True):
    # data_hsi = scio.loadmat(r"E:\code2024\HSI-MSI2\data\HAIDONG\HSI.mat")["HSI"]
    # data_msi = scio.loadmat(r"E:\code2024\HSI-MSI2\data\HAIDONG\MSI.mat")["MSI"]
    # data_traingt = scio.loadmat(r"E:\code2024\HSI-MSI2\data\HAIDONG\MSI_label.mat")["MSI_label"]

    data_hsi = scio.loadmat("E:\code2024\spa and spe\HSI-MSI（data public）\data\Houston2018\data_HS_LR.mat")["data_HS_LR"]
    data_msi = scio.loadmat("E:\code2024\spa and spe\HSI-MSI（data public）\data\Houston2018\data_MS_HR.mat")["data_MS_HR"]
    data_traingt= scio.loadmat(r"E:\code2024\spa and spe\HSI-MSI（data public）\data\Houston2018\TRlabel.mat")["TRlabel"]


    data_hsi = minmax_normalize(data_hsi)
    data_msi = minmax_normalize(data_msi)
    print(data_hsi.shape)
    print(data_msi.shape)
    print(data_traingt.shape)

    train_hsiCube, train_labels, _ = createImgCube(data_hsi, data_traingt, createPosWithoutZero(data_hsi, data_traingt), windowSize=patch_size)
    train_patches, _, _ = createImgCube(data_msi, data_traingt, createPosWithoutZero(data_msi, data_traingt), windowSize=patch_size)

    # Print class distribution before augmentation
    train_labels_counter = Counter(train_labels)
    sorted_train_labels = sorted(train_labels_counter.items())  # 按类别序号排序
    print("Training set samples per class before augmentation:")
    for label, count in sorted_train_labels:
        print(f"Class {label}: {count}")

    X_train, X_val, gt_train, gt_val = splitTrainTestSet(train_hsiCube, train_labels, val_ratio, randomState=128)
    X_train_2, X_val_2, _, _ = splitTrainTestSet(train_patches, train_labels, val_ratio, randomState=128)

    # Print class distribution for train and val set
    gt_train_counter = Counter(gt_train)
    sorted_gt_train = sorted(gt_train_counter.items())  # 按类别序号排序
    gt_val_counter = Counter(gt_val)
    sorted_gt_val = sorted(gt_val_counter.items())  # 按类别序号排序

    print("Training set samples per class after splitting:")
    for label, count in sorted_gt_train:
        print(f"Class {label}: {count}")

    print("Validation set samples per class:")
    for label, count in sorted_gt_val:
        print(f"Class {label}: {count}")

    train_hsiCube, train_patches, train_labels = data_aug(train_hsiCube, train_patches, train_labels)
    X_train, X_val, gt_train, gt_val = splitTrainTestSet(train_hsiCube, train_labels, val_ratio, randomState=128)
    X_train_2, X_val_2, _, _ = splitTrainTestSet(train_patches, train_labels, val_ratio, randomState=128)

    # # 创建包含高光谱和多光谱数据的训练集和验证集

    trainset = TensorDataset1(X_train, X_train_2, gt_train)
    valset = TensorDataset1(X_val, X_val_2, gt_val)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, num_workers=0)


    # # 创建仅包含高光谱数据的训练集和验证集
    HSI_trainset = TensorDataset2(X_train, gt_train)
    HSI_valset = TensorDataset2(X_val, gt_val)
    HSI_train_loader = torch.utils.data.DataLoader(dataset= HSI_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    HSI_val_loader = torch.utils.data.DataLoader(dataset=HSI_valset, batch_size=batch_size, shuffle=False, num_workers=0)



    return train_loader, val_loader, HSI_train_loader, HSI_val_loader
