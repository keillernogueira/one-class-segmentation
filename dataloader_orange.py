import os
import numpy as np

import imageio
from skimage import img_as_float

import torch
from torch.utils import data
import torchvision.transforms as transforms

from data_utils import create_distrib, split_train_test, \
    create_or_load_statistics, normalize_images, data_augmentation


class DataLoaderOrange(data.Dataset):

    def __init__(self, mode, dataset_path, crop_size, stride_size,
                 statistics="own", mean=None, std=None, output_path=None):
        super().__init__()
        assert mode in ['Train', 'Test']

        self.mode = mode
        self.dataset_path = dataset_path
        self.crop_size = crop_size
        self.stride_size = stride_size

        self.data, self.labels = self.load_images()
        if len(self.data) == 0:
            raise RuntimeError('Found 0 samples, please check the data set path')

        self.num_classes = len(np.unique(self.labels[0]))

        self.distrib = self.make_dataset()
        print(self.mode + ' distrib = ', len(self.distrib))

        if statistics == "own" and mean is None and std is None:
            self.mean, self.std = create_or_load_statistics(self.data, self.distrib, self.crop_size,
                                                            self.stride_size, output_path)
        elif statistics == "own" and mean is not None and std is not None:
            self.mean = mean
            self.std = std
        elif statistics == "coco":
            self.mean = np.asarray([0.485, 0.456, 0.406])
            self.std = np.asarray([0.229, 0.224, 0.225])

    def load_images(self):
        first = True
        images = []
        masks = []

        image = None
        for f in os.listdir(self.dataset_path):
            if os.path.isfile(os.path.join(self.dataset_path, f)):
                temp_image = img_as_float(imageio.imread(os.path.join(self.dataset_path, f)))
                if first is True:
                    image = np.expand_dims(temp_image, axis=2)
                    first = False
                else:
                    image = np.concatenate((image, np.expand_dims(temp_image, axis=2)), axis=2)

        # remove NO_DATA values
        image[np.where(image <= -100)] = 0.0

        images.append(image)
        masks.append(imageio.imread(os.path.join(self.dataset_path, 'mask', 'sequoia_raster.tif')))

        return images, masks

    def make_dataset(self):
        all_distrib = create_distrib(self.labels, self.crop_size, self.stride_size, self.num_classes, return_all=False)
        train_distrib, test_distrib = split_train_test(all_distrib, limit=5050)

        if self.mode == 'Train':
            return train_distrib
        else:
            return test_distrib

    def __getitem__(self, index):
        cur_map, cur_x, cur_y = self.distrib[index][0], self.distrib[index][1], self.distrib[index][2]

        img = np.copy(self.data[cur_map][cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size, :])
        label = np.copy(self.labels[cur_map][cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size])

        # Normalization.
        normalize_images(img, self.mean, self.std)

        if self.mode == 'Train':
            img, label = data_augmentation(img, label)

        img = np.transpose(img, (2, 0, 1))

        # Turning to tensors.
        img = torch.from_numpy(img.copy())
        label = torch.from_numpy(label.copy())

        # Returning to iterator.
        return img.float(), label, cur_map, cur_x, cur_y

    def __len__(self):
        return len(self.distrib)
