import os
import numpy as np

from PIL import Image

import imageio
from skimage import img_as_float
from skimage import transform

import torch
from torch.utils import data

from dataloaders.data_utils import create_distrib, create_or_load_statistics, \
    normalize_images, create_distrib_knn, data_augmentation

Image.MAX_IMAGE_PIXELS = None


class DataLoader5Billion(data.Dataset):

    def __init__(self, mode, dataset, dataset_input_path, images, crop_size, stride_size,
                 statistics="own", mean=None, std=None, output_path=None):
        super().__init__()
        assert mode in ['Full_train', 'Full_test']

        self.mode = mode
        self.dataset = dataset
        self.dataset_input_path = dataset_input_path
        self.images = images
        self.crop_size = crop_size
        self.stride_size = stride_size

        self.data, self.labels, self.original_labels = self.load_images()
        self.num_classes = 2

        self.distrib, self.gen_classes = self.make_dataset()

        if statistics == "own" and mean is None and std is None:
            self.mean, self.std = create_or_load_statistics(self.data, self.distrib, self.crop_size,
                                                            self.stride_size, output_path)
        elif statistics == "own" and mean is not None and std is not None:
            self.mean = mean
            self.std = std
        elif statistics == "coco":
            self.mean = np.asarray([0.485, 0.456, 0.406])
            self.std = np.asarray([0.229, 0.224, 0.225])

        if len(self.distrib) == 0:
            raise RuntimeError('Found 0 samples, please check the data set path')

    def convert_mask(self, img):
        img_prepared = np.zeros(img.shape, dtype=int)
        img_prepared[(img == 11) | (img == 13) | (img == 14) | (img == 15)] = 1
        return img_prepared

    def load_images(self):
        images = []
        original_masks = []
        prepared_masks = []
        for img in self.images:
            temp_image = img_as_float(imageio.imread(os.path.join(self.dataset_input_path,
                                                                  'selected_16bit_BGRNIR', img + '.tiff')))
            temp_mask = imageio.imread(os.path.join(self.dataset_input_path, 'annotation',
                                                    img + '_24label.png')).astype(int)
            images.append(temp_image[:, :, 1:4])  # GRNIR
            original_masks.append(temp_mask)
            prepared_masks.append(self.convert_mask(temp_mask))
            print(img, temp_image.shape, temp_mask.shape, np.bincount(temp_mask.flatten()))

        return images, prepared_masks, original_masks

    def make_dataset(self):
        distrib, gen_classes = create_distrib(self.labels, self.crop_size, self.stride_size,
                                              self.num_classes, return_all=True)

        return distrib, gen_classes

    def __getitem__(self, index):
        cur_map, cur_x, cur_y = self.distrib[index][0], self.distrib[index][1], self.distrib[index][2]

        img = np.copy(self.data[cur_map][cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size, :])
        label = np.copy(self.labels[cur_map][cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size])
        original_label = np.copy(self.original_labels[cur_map][cur_x:cur_x + self.crop_size,
                                 cur_y:cur_y + self.crop_size])

        # Normalization.
        normalize_images(img, self.mean, self.std)

        if 'Train' in self.mode or 'train' in self.mode:
            img, label = data_augmentation(img, label)

        img = np.transpose(img, (2, 0, 1))

        # Turning to tensors.
        img = torch.from_numpy(img.copy())
        label = torch.from_numpy(label.copy())

        # Returning to iterator.
        return img.float(), label, cur_map, cur_x, cur_y, original_label

    def __len__(self):
        return len(self.distrib)
