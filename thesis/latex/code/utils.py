import json
import os

from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from skimage.transform import resize
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision


def filedict_from_json(json_path, key):
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
    listdict = json_dict.get(key, [])
    return listdict


class NiiDataset(Dataset):
    def __init__(self, root_dir, img_dirs, transform=None):
        self.root_dir = root_dir
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        merge_data = np.load(os.path.join(self.root_dir, self.img_dirs[idx]))
        image = merge_data[np.newaxis, np.newaxis, ..., 0].astype(np.float32)
        label = merge_data[np.newaxis, np.newaxis, ..., 1].astype(np.float32)
        data = {"data": image, "label": label}
        if self.transform:
            data = self.transform(**data)
        data = {"data": np.squeeze(data["data"], 0), "label": np.squeeze(data["label"], 0)}
        return data


class DiceLoss(nn.Module):
    def __init__(self, num_class, smooth=1):
        super().__init__()
        self.num_class = num_class
        self.smooth = smooth

    def forward(self, inputs, targets):
        dice = 0
        for index in range(self.num_class):
            dice += self.dice_coef(inputs[:, index, ...], targets[:, index, ...])
        return 1 - dice / self.num_class

    def dice_coef(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return dice


class CropResize(AbstractTransform):
    def __init__(self, output_size, data_key="data", label_key="label"):
        self.output_size = output_size
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        image = np.squeeze(data_dict[self.data_key], axis=(0, 1))
        label = np.squeeze(data_dict[self.label_key], axis=(0, 1))

        label_box = self.get_box_index(label)

        image = self.get_image_box(image, label_box)
        label = self.get_image_box(label, label_box)

        image = resize(image, self.output_size)
        label = resize(label, self.output_size, order=0,
                       anti_aliasing=False)

        data_dict[self.data_key] = np.expand_dims(image, (0, 1))
        data_dict[self.label_key] = np.expand_dims(label, (0, 1))

        return data_dict

    @staticmethod
    def get_box_index(label):
        mask_index = np.array((label > 0).nonzero())
        max_index = (mask_index.max(axis=1)).clip(max=label.shape)
        min_index = (mask_index.min(axis=1)).clip(min=0)
        center = np.ceil((max_index + min_index) / 2).astype(int)
        length = np.ceil((max_index - min_index) / 2).max().astype(int)
        pad = int(length * 0.1)
        max_index = (center + length + pad).clip(max=label.shape)
        min_index = (center - length - pad).clip(min=0)
        return max_index, min_index

    @staticmethod
    def get_image_box(image, box):
        max_index, min_index = box
        return image[
               min_index[0]: max_index[0],
               min_index[1]: max_index[1],
               min_index[2]: max_index[2],
               ]


def KL_loss(mean, std):
    return torch.mean(0.5 * (
            torch.sum(torch.pow(std, 2), 1)
            + torch.sum(torch.pow(mean, 2), 1)
            - 2 * torch.sum(torch.log(std + 0.00001), 1)
    ))


class TensorBoardWriter():
    def __init__(self, logdir):
        self.writer = SummaryWriter(logdir=logdir)

    def add_image(self, tag, image, step):
        self.writer.add_image(
            tag,
            torchvision.utils.make_grid(image.detach()),
            step
        )

    def add_scale(self, tag, num, step):
        self.writer.add_scalar(
            tag,
            num,
            step
        )
