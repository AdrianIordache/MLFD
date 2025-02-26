import os
import gc
import cv2
import time
import glob
import copy
import wandb
import pickle
import random
import scipy.io
import numpy as np
import pandas as pd
import polars as pol
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from typing import List
from itertools import cycle
from IPython.display import display
from collections import OrderedDict
from torchvision import datasets, transforms
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR

import timm
import torchvision

import albumentations as A
import torchvision.transforms as T

# from timm.utils import accuracy
from timm.data.auto_augment import rand_augment_transform
from timm.data import create_transform

import warnings
warnings.filterwarnings("ignore", message = "torch.distributed.reduce_op is deprecated")

from labels import *

DECIMALS = 5
VERBOSITY = 30
RD = lambda x: np.round(x, DECIMALS)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sampler = None
def seed_everything(SEED = 42):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(SEED)

def seed_worker(worker_id):
    np.random.seed(SEED)
    random.seed(SEED)

SEED = 42
seed_everything(SEED)

PATH_TO_CIFAR        = './data/cifar-100'
PATH_TO_CIFAR_TRAIN  = "./data/cifar-100/train"
PATH_TO_CIFAR_TEST   = "./data/cifar-100/test"
PATH_TO_CIFAR_META   = "./data/cifar-100/meta"

PATH_TO_TINY_IMAGENET_TRAIN = "./data/tiny-imagenet-200/train.csv"
PATH_TO_TINY_IMAGENET_TEST  = "./data/tiny-imagenet-200/valid.csv"

PATH_TO_IMAGENET_SKETCH_TRAIN = "./data/imagenet-sketch/train.csv"
PATH_TO_IMAGENET_SKETCH_TEST  = "./data/imagenet-sketch/valid.csv"

PATH_TO_CALTECH_TRAIN = "./data/caltech-101/train.csv"
PATH_TO_CALTECH_TEST  = "./data/caltech-101/valid.csv"

PATH_TO_FLOWERS_TRAIN = "./data/flowers-102/train.csv"
PATH_TO_FLOWERS_TEST  = "./data/flowers-102/valid.csv"

PATH_TO_CUB200_TRAIN = "./data/cub-200-2011/train.csv"
PATH_TO_CUB200_TEST  = "./data/cub-200-2011/valid.csv"

PATH_TO_PETS_TRAIN   = "./data/oxford-pets/train.csv"
PATH_TO_PETS_TEST    = "./data/oxford-pets/valid.csv"

CIFAR_MEANS = [0.5073620348243464, 0.4866895632914624, 0.44108857134650736]
CIFAR_STDS  = [0.2674881548800154, 0.2565930997269334, 0.2763085095510782]

IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS  = [0.229, 0.224, 0.225]

def time_since(since, percent):
    def seconds_as_minutes(seconds):
        import math
        minutes  = math.floor(seconds / 60)
        seconds -= minutes * 60
        return f'{int(minutes)}m {int(seconds)}s'

    now     = time.time()
    seconds = now - since

    total_seconds    = seconds / (percent)
    remained_seconds = total_seconds - seconds
    return f'{seconds_as_minutes(seconds)} ({seconds_as_minutes(remained_seconds)})'

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def update(self, value, n = 1):
        self.count += n
        self.sum   += value * n

        self.value   = value 
        self.average = self.sum / self.count

    def reset(self):
        self.value, self.average, self.sum, self.count = 0, 0, 0, 0

def overfit_the_batch(loader):
    batch = 49
    iterator = iter(loader)
    images, labels = next(iterator)
    while labels.sum() == 0:
        images, labels = next(iterator)    
    
    # images = torch.tensor(np.zeros((16, 3, 256, 256))).float().to(device)
    return batch, images, labels

def reverse_normalization(images, means = CIFAR_MEANS, stds = CIFAR_STDS):
    reversed_images = images.new(*images.size())
    reversed_images[:, 0, :, :] = images[:, 0, :, :] * stds[0] + means[0]
    reversed_images[:, 1, :, :] = images[:, 1, :, :] * stds[1] + means[1]
    reversed_images[:, 2, :, :] = images[:, 2, :, :] * stds[2] + means[2]
    return reversed_images


def imshow(inp, title = None, means = IMAGENET_MEANS, stds = IMAGENET_STDS):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(means)
    std = np.array(stds)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp[:,:,[2,1,0]])
    # plt.savefig("image.png")
    plt.show()

    # inputs, classes = next(iter(trainloader))
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out, title = [x.item() for x in classes])

def debug_tests():
    batch, images, labels = overfit_the_batch(loader)
    reversed_images = reverse_normalization(images)
    plt.imshow(reversed_images[1].permute(1, 2, 0))
    plt.show()

def free_gpu_memory(device, object = None, verbose = False):
    if object == None:
        for object in gc.get_objects():
            try:
                if torch.is_tensor(object) or (hasattr(object, 'data' and  torch.is_tensor(object.data))):
                    if verbose: print(f"GPU Memory Used: {object}, with size: {object.size()}")
                    object = object.detach().cpu()
                    del object
            except:
                pass
    else:
        object = object.detach().cpu()
        del object

    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

def generate_folds(data: pd.DataFrame, skf_column: str, n_folds: int = 5, random_state = SEED):
    skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = SEED)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(data, data[skf_column])):
        data.loc[valid_idx, 'fold'] = fold

    data['fold'] = data['fold'].astype(int)
    return data

def get_optimizer(parameters, config_file):
    assert config_file["optimizer"] in \
        ["SGD", "Adam", "AdamW"], "[optimizer]: Option not implemented"

    if config_file["optimizer"] == "SGD":
        return optim.SGD(parameters, **config_file["optimizer_param"])
    if config_file["optimizer"] == "Adam":
        return optim.Adam(parameters, **config_file["optimizer_param"])
    if config_file["optimizer"] == "AdamW":
        return optim.AdamW(parameters, **config_file["optimizer_param"])

    return None


def get_scheduler(optimizer, config_file = None):
    if config_file['scheduler'] == 'CosineAnnealingWarmRestarts':
        return CosineAnnealingWarmRestarts(
            optimizer, **config_file["scheduler_param"]
        )
    if config_file['scheduler'] == 'OneCycleLR':
        return OneCycleLR(
            optimizer, **config_file["scheduler_param"]
        )
    if config_file['scheduler'] == 'CosineAnnealingLR':
        return CosineAnnealingLR(
            optimizer, **config_file["scheduler_param"]
        )
        
    return None

def load_cifar(path):
    with open(path, "rb") as file:
        data_dict = pickle.load(file, encoding = "bytes")

    images = data_dict[b'data']
    labels = np.array(data_dict[b'fine_labels'])
    return images, labels

def compute_mean_and_std(train_images, test_images):
    cifar_images = np.vstack((train_images, test_images))

    mean = [
        np.mean(cifar_images[:,      : 1024], axis = (0, 1)) / 255,    
        np.mean(cifar_images[:, 1024 : 2048], axis = (0, 1)) / 255,
        np.mean(cifar_images[:, 2048 :     ], axis = (0, 1)) / 255
    ]

    std  = [
        np.std(cifar_images[:,      : 1024], axis = (0, 1)) / 255,    
        np.std(cifar_images[:, 1024 : 2048], axis = (0, 1)) / 255,
        np.std(cifar_images[:, 2048 :     ], axis = (0, 1)) / 255
    ]

    return mean, std

def prep_tiny_imagenet():
    PATH_TO_TINY_IMAGENET_TRAIN  = "./data/tiny-imagenet-200/train"
    PATH_TO_TINY_IMAGENET_VALID  = "./data/tiny-imagenet-200/val"
    PATH_TO_TINY_IMAGENET_LABELS = "./data/tiny-imagenet-200/words.txt"

    # Generate train data csv
    lines = dict()
    with open(PATH_TO_TINY_IMAGENET_LABELS, "r") as handler:
        line = handler.readline()
        while line:
            label, text  = line.strip("\n").split("\t")
            lines[label] = text
            line  = handler.readline()

    label_to_idx    = {}
    train_label_map = {
        "path"  : [],
        "class" : [],
        "label" : [],
        "text"  : [],
    }

    for idx, folder in enumerate(glob.glob(PATH_TO_TINY_IMAGENET_TRAIN + "/*")):
        label       = folder.split("/")[-1]
        description = lines[label]
        
        label_to_idx[label] = (idx, description) 
        for image_path in glob.glob(f"{folder}/images/*.JPEG"):
            train_label_map["path"].append(image_path)
            train_label_map["class"].append(idx)
            train_label_map["label"].append(label)
            train_label_map["text"].append(description)

    train_label_map = pd.DataFrame(train_label_map)
    train_label_map = train_label_map.sample(frac = 1, random_state = SEED)
    train_label_map.reset_index(drop = True, inplace = True)
    train_label_map.to_csv("./data/tiny-imagenet-200/train.csv", index = False)
    display(train_label_map)

    # Generate valid data csv
    valid_label_map = {
        "path"  : [],
        "class" : [],
        "label" : [],
        "text"  : [],
    }

    with open(PATH_TO_TINY_IMAGENET_VALID + "/val_annotations.txt", "r") as handler:
        line = handler.readline()
        while line:
            name       = line.strip("\n").split("\t")[0]
            image_path = f"{PATH_TO_TINY_IMAGENET_VALID}/images/{name}"

            label = line.strip("\n").split("\t")[1]
            (class_id, desc) = label_to_idx[label]

            valid_label_map["path"].append(image_path)
            valid_label_map["class"].append(class_id)
            valid_label_map["label"].append(label)
            valid_label_map["text"].append(desc)

            line  = handler.readline()

    valid_label_map = pd.DataFrame(valid_label_map)
    valid_label_map = valid_label_map.sample(frac = 1, random_state = SEED)
    valid_label_map.reset_index(drop = True, inplace = True)
    valid_label_map.to_csv("./data/tiny-imagenet-200/valid.csv", index = False)
    display(valid_label_map)

def prep_imagenet_sketch():
    PATH_TO_IMAGENET_SKETCH_IMAGES = "./data/imagenet-sketch/images"

    label_map = {
        "path"  : [],
        "class" : [],
        "label" : [],
        "text"  : [],
    }

    for class_idx, directory in enumerate(glob.glob(PATH_TO_IMAGENET_SKETCH_IMAGES + "/*")):
        label = directory.split("/")[-1] 
        text  = IMAGENET_LABELS[label]

        for image_path in glob.glob(directory + "/*"):
            label_map["path"].append(image_path)
            label_map["class"].append(class_idx)
            label_map["label"].append(label)
            label_map["text"].append(text)


    label_map = pd.DataFrame(label_map)
    label_map = label_map.sample(frac = 1, random_state = SEED)
    label_map.reset_index(drop = True, inplace = True)
    label_map = generate_folds(label_map, "class")
    # label_map["train"] = label_map["fold"].apply(lambda x: True if x < 4 else False)

    train_data = label_map[label_map["fold"]  < 4].reset_index(drop = True, inplace = False)
    valid_data = label_map[label_map["fold"] == 4].reset_index(drop = True, inplace = False)
    
    train_data.to_csv("./data/imagenet-sketch/train.csv", index = False)
    valid_data.to_csv("./data/imagenet-sketch/valid.csv", index = False)
    
    display(train_data)
    display(valid_data)

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


class AdaptedCELoss(nn.Module):
    def __init__(self):
        super(AdaptedCELoss, self).__init__()

    def forward(self, soft_prob, soft_targets):
        return -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0]

def get_activation(name, out_dict):
    def hook(model, input, output):
        out_dict[name] = output
    return hook


def prep_caltech():
    PATH_TO_CALTECH = "./data/caltech-101/"

    label_map = {
        "path"  : [],
        "class" : [],
        "label" : []
    }

    for idx, folder in enumerate(glob.glob(PATH_TO_CALTECH + "*")):
        label = folder.split("/")[-1]

        for image_path in glob.glob(f"{folder}/*.jpg"):
            label_map["path"].append(image_path)
            label_map["class"].append(idx)
            label_map["label"].append(label)

            # break

    label_map = pd.DataFrame(label_map)
    label_map = label_map.sample(frac = 1, random_state = SEED)
    label_map.reset_index(drop = True, inplace = True)

    # display(len(np.unique(label_map["class"].values)))
    train, valid = train_test_split(label_map, test_size = 0.2, random_state = SEED, stratify = label_map["class"])
    # train.to_csv("./data/caltech-101/train.csv", index = False)
    # valid.to_csv("./data/caltech-101/valid.csv", index = False)


def prep_flowers():
    PATH_TO_FLOWERS_IMAGES = "./data/flowers-102/images/"
    PATH_TO_FLOWERS_LABELS = "./data/flowers-102/imagelabels.mat"
    PATH_TO_FLOWERS_FOLDS  = "./data/flowers-102/setid.mat"

    labels = scipy.io.loadmat(PATH_TO_FLOWERS_LABELS)
    setids = scipy.io.loadmat(PATH_TO_FLOWERS_FOLDS)

    labels    = labels["labels"][0]
    train_idx = setids["trnid"][0].tolist()
    valid_idx = setids["valid"][0].tolist()
    test_idx  = setids["tstid"][0].tolist()

    image_id_to_label = dict(enumerate((labels - 1).tolist(), 1))
    print(len(np.unique(labels)))
    train_label_map = {
        "path"  : [],
        "class" : []
    }


    training = train_idx + test_idx
    for idx in training:
        train_label_map["path"].append(PATH_TO_FLOWERS_IMAGES + f"image_{idx:05d}.jpg")
        train_label_map["class"].append(image_id_to_label[idx])

    train_label_map = pd.DataFrame(train_label_map)
    train_label_map = train_label_map.sample(frac = 1, random_state = SEED)
    train_label_map.reset_index(drop = True, inplace = True)
    # train_label_map.to_csv("./data/flowers-102/train.csv", index = False)
    display(train_label_map)

    valid_label_map = {
        "path"  : [],
        "class" : []
    }

    for idx in valid_idx:
        valid_label_map["path"].append(PATH_TO_FLOWERS_IMAGES + f"image_{idx:05d}.jpg")
        valid_label_map["class"].append(image_id_to_label[idx])

    valid_label_map = pd.DataFrame(valid_label_map)
    valid_label_map = valid_label_map.sample(frac = 1, random_state = SEED)
    valid_label_map.reset_index(drop = True, inplace = True)
    # valid_label_map.to_csv("./data/flowers-102/valid.csv", index = False)
    display(valid_label_map)

def prep_cub200():
    PATH_TO_CUB_IMAGES = "./data/cub-200-2011/images/"

    label_map = {
        "path"  : [],
        "class" : [],
        "label" : []
    }

    for idx, folder in enumerate(glob.glob(PATH_TO_CUB_IMAGES + "*")):
        label = folder.split("/")[-1]

        for image_path in glob.glob(f"{folder}/*.jpg"):
            label_map["path"].append(image_path)
            label_map["class"].append(idx)
            label_map["label"].append(label)

            # break

    label_map = pd.DataFrame(label_map)
    label_map = label_map.sample(frac = 1, random_state = SEED)
    label_map.reset_index(drop = True, inplace = True)

    # display(label_map)
    # display(len(np.unique(label_map["class"].values)))
    train, valid = train_test_split(label_map, test_size = 0.2, random_state = SEED, stratify = label_map["class"])
    train.to_csv("./data/cub-200-2011/train.csv", index = False)
    valid.to_csv("./data/cub-200-2011/valid.csv", index = False)

def prep_cars():
    PATH_TO_CARS_TRAIN_IMAGES = "./data/stanford-cars/car_data/car_data/train/"
    PATH_TO_CARS_TEST_IMAGES  = "./data/stanford-cars/car_data/car_data/test/" 
    PATH_TO_CARS_TRAIN_LABELS = "./data/stanford-cars/anno_train.csv"
    PATH_TO_CARS_TEST_LABELS  = "./data/stanford-cars/anno_test.csv"

    label_to_idx    = {}
    train_label_map = {
        "path"  : [],
        "class" : [],
        "label" : []
    }

    for idx, folder in sorted(enumerate(glob.glob(PATH_TO_CARS_TRAIN_IMAGES + "*"))):
        label = folder.split("/")[-1]

        label_to_idx[label] = idx
        for image_path in glob.glob(f"{folder}/*.jpg"):
            train_label_map["path"].append(image_path)
            train_label_map["class"].append(idx)
            train_label_map["label"].append(label)

    train_label_map = pd.DataFrame(train_label_map)
    train_label_map = train_label_map.sample(frac = 1, random_state = SEED)
    train_label_map.reset_index(drop = True, inplace = True)

    train_label_map.to_csv("./data/stanford-cars/train.csv", index = False)

    valid_label_map = {
        "path"  : [],
        "class" : [],
        "label" : []
    }

    for idx, folder in sorted(enumerate(glob.glob(PATH_TO_CARS_TEST_IMAGES + "*"))):
        label = folder.split("/")[-1]

        for image_path in glob.glob(f"{folder}/*.jpg"):
            valid_label_map["path"].append(image_path)
            valid_label_map["class"].append(label_to_idx[label])
            valid_label_map["label"].append(label)

    valid_label_map = pd.DataFrame(valid_label_map)
    valid_label_map = valid_label_map.sample(frac = 1, random_state = SEED)
    valid_label_map.reset_index(drop = True, inplace = True)

    valid_label_map.to_csv("./data/stanford-cars/valid.csv", index = False)

    # train_labels = pd.read_csv(PATH_TO_CARS_TRAIN_LABELS, names = ["id", "box1", "box2", "box3", "box4", "class"])
    # train_labels["path"]  = train_labels["id"].apply(lambda x: PATH_TO_CARS_TRAIN_IMAGES + f"{x}")
    # train_labels["class"] = train_labels["class"].apply(lambda x: x - 1)
    # train_labels = train_labels[['id', 'path', 'class']]
    # print(np.unique(train_labels["class"]))
    # display(train_labels)

    # print(len(glob.glob(PATH_TO_CARS_TEST_IMAGES + "*.jpg")))
    # test_labels = pd.read_csv(PATH_TO_CARS_TEST_LABELS, names = ["id", "box1", "box2", "box3", "box4", "class"])
    # test_labels["path"]  = test_labels["id"].apply(lambda x: PATH_TO_CARS_TEST_IMAGES + f"{x}")
    # test_labels["class"] = test_labels["class"].apply(lambda x: x - 1)
    # test_labels = test_labels[['id', 'path', 'class']]
    # display(test_labels)
    # # print(labels["annotations"][0][0])

    # train_labels.to_csv("./data/stanford-cars/train.csv", index = False)
    # test_labels.to_csv("./data/stanford-cars/valid.csv", index = False)


def prep_pets():
    from sklearn.preprocessing import OrdinalEncoder
    PATH_TO_PETS_IMAGES = "./data/oxford-pets/images/"

    label_map = {
        "path"  : [],
        "label" : []
    }
    
    for image_path in sorted(glob.glob(PATH_TO_PETS_IMAGES + "*.jpg")):
        label = image_path.split("/")[-1].split(".")[0]
        label = "_".join(label.split("_")[:-1]).lower()
        # print(label)

        image = cv2.imread(image_path)
        if image is None:
            print(image_path)
        else:
            label_map["path"].append(image_path)
            label_map["label"].append(label)

    label_map = pd.DataFrame(label_map)
    label_map = label_map.sample(frac = 1, random_state = SEED)
    label_map.reset_index(drop = True, inplace = True)

    encoder = OrdinalEncoder()
    label_map["class"] = encoder.fit_transform(label_map["label"].values.reshape(-1, 1))
    label_map["class"] = label_map["class"].astype(int)
    # print(np.unique(label_map["class"]))
    display(label_map)

    train, valid = train_test_split(label_map, test_size = 0.2, random_state = SEED, stratify = label_map["class"])
    train.to_csv("./data/oxford-pets/train.csv", index = False)
    valid.to_csv("./data/oxford-pets/valid.csv", index = False)
