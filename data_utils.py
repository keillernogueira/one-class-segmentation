import os
import random
from skimage import transform

import scipy
import numpy as np


def create_distrib(labels, crop_size, stride_size, num_classes, return_all=False):
    instances = [[[] for i in range(0)] for i in range(num_classes)]
    counter = num_classes * [0]
    binc = np.zeros((num_classes, num_classes))  # cumulative bincount for each class

    for k in range(0, len(labels)):
        w, h = labels[k].shape
        for i in range(0, w, stride_size):
            for j in range(0, h, stride_size):
                cur_map = k
                cur_x = i
                cur_y = j
                patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if len(patch_class) != crop_size and len(patch_class[0]) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                assert patch_class.shape == (crop_size, crop_size), \
                    "Error create_distrib: Current patch size is " + str(len(patch_class)) + "x" + str(len(patch_class[0]))

                count = np.bincount(patch_class.astype(int).flatten(), minlength=2)
                if count[1] != 0:
                    # if count[1] > percentage_pos_class * count[0]:
                    instances[1].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                    counter[1] += 1
                    binc[1] += count
                else:
                    instances[0].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                    counter[0] += 1
                    binc[0] += count

    for i in range(len(counter)):
        print('Class ' + str(i) + ' has length ' + str(counter[i]) + ' - ' + np.array_str(binc[i]).replace("\n", ""))

    if return_all:
        return np.asarray(instances[0] + instances[1])
    else:
        return np.asarray(instances[1])


def create_distrib_knn(labels, crop_size, stride_size, num_classes):
    instances = []
    counter = 0
    binc = np.zeros(num_classes)

    for k in range(0, len(labels)):
        w, h = labels[k].shape
        for i in range(0, w, stride_size):
            for j in range(0, h, stride_size):
                cur_map = k
                cur_x = i
                cur_y = j
                patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if len(patch_class) != crop_size and len(patch_class[0]) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                assert patch_class.shape == (crop_size, crop_size), \
                    "Error create_distrib: Current patch size is " + str(len(patch_class)) + "x" + str(len(patch_class[0]))

                count = np.bincount(patch_class.astype(int).flatten(), minlength=2)
                proportion = count[1]/float(np.sum(count))
                # if 0.45 <= proportion <= 0.55:
                if 0.35 <= proportion <= 0.65:
                    if counter < 16:
                        counter += 1
                        binc += count
                        instances.append((cur_map, cur_x, cur_y, count, proportion))
                    else:
                        # find patch with largest value
                        index_patch_max_pos = -1
                        patch_max_value = -999999999
                        for x in range(len(instances)):
                            if abs(0.5 - instances[x][4]) > patch_max_value:
                                patch_max_value = abs(0.5 - instances[x][4])
                                index_patch_max_pos = x
                        # exchange if it is the case
                        if abs(0.5 - instances[index_patch_max_pos][4]) > abs(0.5 - proportion):
                            binc -= instances[index_patch_max_pos][3]
                            del instances[index_patch_max_pos]
                            instances.append((cur_map, cur_x, cur_y, count, proportion))
                            binc += count

    print('Number samples ' + str(counter) + ' - ' + np.array_str(binc).replace("\n", ""))

    return np.asarray(instances)


def split_train_test(data_distribution, limit=5050):
    train_distrib = []
    test_distrib = []
    for el in data_distribution:
        if el[1] > limit:
            test_distrib.append(el)
        else:
            train_distrib.append(el)

    return np.asarray(train_distrib), np.asarray(test_distrib)


def normalize_images(data, _mean, _std):
    for i in range(len(_mean)):
        data[:, :, i] = np.subtract(data[:, :, i], _mean[i])
        data[:, :, i] = np.divide(data[:, :, i], _std[i])


def compute_image_mean(data):
    _mean = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    _std = np.std(np.std(np.std(data, axis=0, ddof=1), axis=0, ddof=1), axis=0, ddof=1)

    return _mean, _std


def dynamically_calculate_mean_and_std(data, distrib, crop_size):
    mean_full = []
    std_full = []

    all_patches = []

    for i in range(len(distrib)):
        cur_map = distrib[i][0]
        cur_x = distrib[i][1]
        cur_y = distrib[i][2]
        patch = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

        if len(patch[0]) != crop_size or len(patch[1]) != crop_size:
            raise NotImplementedError("Error! Current patch size: " +
                                      str(len(patch)) + "x" + str(len(patch[0])))

        all_patches.append(patch)

        if i > 0 and i % 5000 == 0:
            mean, std = compute_image_mean(np.asarray(all_patches))
            mean_full.append(mean)
            std_full.append(std)
            all_patches = []

    # remaining images
    print(np.asarray(all_patches).shape)
    mean, std = compute_image_mean(np.asarray(all_patches))
    mean_full.append(mean)
    std_full.append(std)

    return np.mean(mean_full, axis=0), np.mean(std_full, axis=0)


def create_or_load_statistics(data, distrib, crop_size, stride_size, output_path):
    # create mean, std from training
    if os.path.isfile(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                      str(stride_size) + '_mean.npy')):
        _mean = np.load(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                                     str(stride_size) + '_mean.npy'), allow_pickle=True)
        _std = np.load(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                                    str(stride_size) + '_std.npy'), allow_pickle=True)
    else:
        _mean, _std = dynamically_calculate_mean_and_std(data, distrib, crop_size)
        np.save(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                str(stride_size) + '_mean.npy'), _mean)
        np.save(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                str(stride_size) + '_std.npy'), _std)
    print(_mean, _std)
    return _mean, _std


def data_augmentation(img, label):
    rand_fliplr = np.random.random() > 0.50
    rand_flipud = np.random.random() > 0.50
    rand_rotate = np.random.random()

    if rand_fliplr:
        img = np.fliplr(img)
        label = np.fliplr(label)
    if rand_flipud:
        img = np.flipud(img)
        label = np.flipud(label)

    if rand_rotate < 0.25:
        img = transform.rotate(img, 270, order=1, preserve_range=True)
        label = transform.rotate(label, 270, order=0, preserve_range=True)
    elif rand_rotate < 0.50:
        img = transform.rotate(img, 180, order=1, preserve_range=True)
        label = transform.rotate(label, 180, order=0, preserve_range=True)
    elif rand_rotate < 0.75:
        img = transform.rotate(img, 90, order=1, preserve_range=True)
        label = transform.rotate(label, 90, order=0, preserve_range=True)

    img = img.astype(np.float32)
    label = label.astype(np.int64)

    return img, label
