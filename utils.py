import os
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import decomposition
from sklearn.manifold import TSNE
# from tsne_torch import TorchTSNE as TSNE

import torch
from torch import nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_best_models(net, optimizer, output_path, best_records, epoch, acc, acc_cls, cm):
    if len(best_records) < 5:
        best_records.append({'epoch': epoch, 'acc': acc, 'acc_cls': acc_cls, 'cm': cm})

        torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
    else:
        # find min saved acc
        min_index = 0
        for i, r in enumerate(best_records):
            if best_records[min_index]['acc_cls'] > best_records[i]['acc_cls']:
                min_index = i
        # check if currect acc is greater than min saved acc
        if acc_cls > best_records[min_index]['acc_cls']:
            # if it is, delete previous files
            min_step = str(best_records[min_index]['epoch'])

            os.remove(os.path.join(output_path, 'model_' + min_step + '.pth'))
            os.remove(os.path.join(output_path, 'opt_' + min_step + '.pth'))

            # replace min value with current
            best_records[min_index] = {'epoch': epoch, 'acc': acc, 'acc_cls': acc_cls, 'cm': cm}

            # save current model
            torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))


def project_data(data, labels, pca_n_components):
    print('data.shape', data.shape)
    pca_model = decomposition.PCA(n_components=pca_n_components, random_state=12345)
    pca_data = pca_model.fit_transform(data)
    print('pca_data.shape', pca_data.shape)
    tsne_model = TSNE(n_components=2, n_jobs=-1, learning_rate='auto', init='random')
    tsne_data = tsne_model.fit_transform(pca_data)
    print('tsne_data.shape', tsne_data.shape)
    # np.save(os.path.join('tsne_data.npy'), tsne_data)
    # np.save(os.path.join('labels.npy'), labels)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=labels,
                    palette=sns.color_palette("hls", 2), legend="full", alpha=0.3)
    # plt.show()
    plt.savefig('plot.png')
