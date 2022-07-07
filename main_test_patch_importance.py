import imageio
import numpy as np

from PIL import Image, ImageDraw


def create_distrib(labels, crop_size=128, stride_size=64, num_classes=2, dataset='River', return_all=True):
    instances = [[[] for i in range(0)] for i in range(num_classes)]
    counter = num_classes * [0]
    binc = np.zeros((num_classes, num_classes))  # cumulative bincount for each class
    gen_classes = []

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
                if dataset == 'Coffee':
                    if count[1] >= count[0]:
                        instances[1].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                        gen_classes.append(1)
                        counter[1] += 1
                        binc[1] += count
                    else:
                        instances[0].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                        gen_classes.append(0)
                        counter[0] += 1
                        binc[0] += count
                elif dataset == 'Coffee_Full':
                    if len(count) == 2:  # there is only coffee and/or non-coffee
                        if count[1] != 0:  # there is at least one coffee pixel
                            instances[1].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                            gen_classes.append(1)
                            counter[1] += 1
                            binc[1] += count
                        else:
                            instances[0].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                            gen_classes.append(0)
                            counter[0] += 1
                            binc[0] += count
                    else:  # there is background (class 2)
                        if count[2] <= count[0] + count[1]:
                            if count[1] != 0:
                                instances[1].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                                gen_classes.append(1)
                                counter[1] += 1
                                binc[1] += count[0:2]
                            else:
                                instances[0].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                                gen_classes.append(0)
                                counter[0] += 1
                                binc[0] += count[0:2]
                else:
                    # dataset River and Orange
                    if count[1] != 0:
                        # if count[1] > percentage_pos_class * count[0]:
                        instances[1].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                        gen_classes.append(1)
                        counter[1] += 1
                        binc[1] += count[0:2]  # get only the first two positions
                    else:
                        instances[0].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                        gen_classes.append(0)
                        counter[0] += 1
                        binc[0] += count[0:2]  # get only the first two positions

    for i in range(len(counter)):
        print('Class ' + str(i) + ' has length ' + str(counter[i]) + ' - ' + np.array_str(binc[i]).replace("\n", ""))

    if return_all:
        return np.asarray(instances[0] + instances[1]), np.asarray(gen_classes)
    else:
        # this generates an error because len(gen_classes) > len(instances[1])
        # Not using this because training with full training and validation given the weight sampler
        return np.asarray(instances[1]), np.asarray(gen_classes)


def create_distrib_current(labels, crop_size=128, stride_size=64, num_classes=2, dataset='River', return_all=True):
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
                if dataset == 'Coffee':
                    if count[1] >= count[0]:
                        instances[1].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                        counter[1] += 1
                        binc[1] += count
                    else:
                        instances[0].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                        counter[0] += 1
                        binc[0] += count
                elif dataset == 'Coffee_Full':
                    if len(count) == 2:  # there is only coffee and/or non-coffee
                        if count[1] != 0:  # there is at least one coffee pixel
                            instances[1].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                            counter[1] += 1
                            binc[1] += count
                        else:
                            instances[0].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                            counter[0] += 1
                            binc[0] += count
                    else:  # there is background (class 2)
                        if count[2] <= count[0] + count[1]:
                            if count[1] != 0:
                                instances[1].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                                counter[1] += 1
                                binc[1] += count[0:2]
                            else:
                                instances[0].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                                counter[0] += 1
                                binc[0] += count[0:2]
                else:
                    # dataset River and Orange
                    if count[1] != 0:
                        # if count[1] > percentage_pos_class * count[0]:
                        instances[1].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                        counter[1] += 1
                        binc[1] += count[0:2]  # get only the first two positions
                    else:
                        instances[0].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                        counter[0] += 1
                        binc[0] += count[0:2]  # get only the first two positions

    for i in range(len(counter)):
        print('Class ' + str(i) + ' has length ' + str(counter[i]) + ' - ' + np.array_str(binc[i]).replace("\n", ""))

    if return_all:
        # original: bug because order of the gen_classes is different from order of the patches
        # return np.asarray(instances[0] + instances[1]), np.asarray(gen_classes)
        return np.asarray(instances[0] + instances[1]), \
               np.concatenate((np.zeros(len(instances[0]), dtype=int), np.ones(len(instances[1]), dtype=int)))
    else:
        # Not using second return because training with full training and validation given the weight sampler
        return np.asarray(instances[1]), np.asarray([])


if __name__ == '__main__':
    crop_size = 128

    masks = []
    no = 'C:\\Users\\keill\\Desktop\\Datasets\\Bacia_SantoAntonio_Revisado\\new\\' \
         'Rios_editados_Ana_Paula_Keiller\\Rios_Estreitos\\no_mask.tif'
    no_img = imageio.imread(no).astype(int)
    source_no_pil = Image.fromarray(no_img).convert("RGBA")
    no_pil = ImageDraw.Draw(source_no_pil)
    print(no_img.shape, np.bincount(no_img.flatten()))
    masks.append(no_img)

    so = 'C:\\Users\\keill\\Desktop\\Datasets\\Bacia_SantoAntonio_Revisado\\new\\' \
         'Rios_editados_Ana_Paula_Keiller\\Rios_Estreitos\\so_mask.tif'
    so_img = imageio.imread(so).astype(int)
    source_so_pil = Image.fromarray(so_img).convert("RGBA")
    so_pil = ImageDraw.Draw(source_so_pil)
    print(type(so_img), so_img.shape, np.bincount(so_img.flatten()))
    masks.append(so_img)

    distri, gen_classes = create_distrib(masks)
    class_loader_weights = 1. / np.bincount(gen_classes)
    print(distri.shape, gen_classes.shape, np.bincount(gen_classes), class_loader_weights)
    select_ones = distri[np.where(gen_classes == 1)]
    for s in select_ones:  # (cur_map, cur_x, cur_y, np.bincount(patch_class.flatten()))
        if s[0] == 0:
            no_pil.rectangle(((s[1], s[2]), (s[1]+crop_size, s[2]+crop_size)), outline="red", width=2)
        else:
            so_pil.rectangle(((s[1], s[2]), (s[1]+crop_size, s[2]+crop_size)), outline="red", width=2)

    source_no_pil.save('C:\\Users\\keill\\Desktop\\no_draw.png', "PNG")
    source_so_pil.save('C:\\Users\\keill\\Desktop\\so_draw.png', "PNG")
    # np.savetxt('C:\\Users\\keill\\Desktop\\t.txt', select_ones, fmt='%s %s %s %s')

    # distri, gen_classes = create_distrib_current(masks)
    # print(distri.shape, gen_classes.shape, np.bincount(gen_classes), gen_classes[0:100])
    # select_ones = distri[np.where(gen_classes == 1)]
    # np.savetxt('C:\\Users\\keill\\Desktop\\t_c.txt', select_ones, fmt='%s %s %s %s')
