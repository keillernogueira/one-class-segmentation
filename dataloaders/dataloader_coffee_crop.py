import os
import numpy as np

import imageio
from skimage import img_as_float

import torch
from torch.utils import data

from dataloaders.data_utils import create_distrib, split_train_test, \
    create_or_load_statistics, normalize_images, data_augmentation


class DataLoaderCoffeeCrop(data.Dataset):
    def __init__(self, mode, dataset, dataset_path, images, crop_size, stride_size,
                 statistics="own", mean=None, std=None, output_path=None):
        super().__init__()
        assert mode in ['Train', 'Test']

        self.mode = mode
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.images = images
        self.crop_size = crop_size
        self.stride_size = stride_size

        self.data, self.labels = self.load_images()
        print(self.data.shape, self.labels.shape)

        if len(self.data) == 0:
            raise RuntimeError('Found 0 samples, please check the data set path')

        self.num_classes = len(np.unique(self.labels[0]))

        self.distrib, self.gen_classes = self.make_dataset()
        print('self.distrib ', len(self.distrib))

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
        images = []
        masks = []
        for img in self.images:
            temp_image = img_as_float(imageio.imread(os.path.join(self.dataset_path, 'crop_' + str(img) + '.tif')))
            temp_mask = imageio.imread(os.path.join(self.dataset_path, 'mask_' + str(img) + '.tif'))
            images.append(temp_image)
            temp_mask[np.where(temp_mask < 128)] = 0
            temp_mask[np.where(temp_mask >= 128)] = 1
            masks.append(temp_mask)

        return np.asarray(images), np.asarray(masks)

    def make_dataset(self):
        distrib, gen_classes = create_distrib(self.labels, self.crop_size, self.stride_size, self.num_classes,
                                              self.dataset, return_all=True)
        return distrib, gen_classes

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
