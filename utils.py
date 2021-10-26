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


def project_data(data, labels, save_name, pca_n_components=50):
    pca_model = decomposition.PCA(n_components=pca_n_components, random_state=12345)
    pca_data = pca_model.fit_transform(data)
    tsne_model = TSNE(n_components=2, n_jobs=-1, learning_rate='auto', init='random')
    tsne_data = tsne_model.fit_transform(pca_data)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=labels,
                    palette=sns.color_palette("hls", 2), legend="full", alpha=0.3)
    # plt.show()
    plt.savefig(save_name)


def get_triples(feat, labs, track_mean):
    feat_pos_lbl = feat[labs == 1, :]  # .detach()
    # feat_neg_lbl = feat[labs == 0, :].detach()
    mean_pos_lbl = torch.mean(feat_pos_lbl, dim=0)
    if track_mean is None:
        track_mean = mean_pos_lbl  # .detach().cpu().numpy()
    else:
        track_mean = torch.mean(torch.stack((track_mean, mean_pos_lbl), dim=0), dim=0)

    cdist_neg_lbl = torch.cdist(torch.unsqueeze(track_mean, dim=0), feat[labs == 0, :], p=2).squeeze()  # .detach()
    _, topk_neg = torch.topk(cdist_neg_lbl, torch.count_nonzero(labs), largest=False)

    return torch.unsqueeze(track_mean, dim=0).repeat(feat_pos_lbl.size(0), 1), \
           feat_pos_lbl, \
           torch.index_select(feat, 0, topk_neg)


def calc_accuracy_triples(a, p, n):
    a_p_dist = torch.cdist(a[0:1, :], p, p=2)
    a_n_dist = torch.cdist(a[0:1, :], n, p=2)
    dist = a_n_dist - 1 > a_p_dist

    return torch.count_nonzero(dist) / a.size(0)


# https://www.kaggle.com/ashishpatel26/triplet-loss-network-for-humpback-whale-prediction
def predict_map(occur_im, feat_flat, track_mean, maps, curxs, curys):
    # print('predict_map', occur_im.shape, feat_flat.shape, track_mean.shape, maps.shape, curxs.shape, curys.shape)
    # predict_map (14329, 13362, 2) (16, 64, 64, 2560) (2560,) torch.Size([16]) torch.Size([16]) torch.Size([16])
    m, h, w, f = feat_flat.shape
    feat_flat = feat_flat.reshape((-1, f))
    dist = (np.linalg.norm(track_mean - feat_flat, axis=1) < 1).astype(np.uint32).reshape((m, h, w))
    # print('1', feat_flat.shape, dist.shape, track_mean.shape)
    # 1 (65536, 2560) (65536,) (2560,)

    for i in range(len(maps)):
        cur_x = curxs[i]
        cur_y = curys[i]

        occur_im[cur_x:cur_x + h, cur_y:cur_y + w, 0] += (1-dist[i, :, :])
        occur_im[cur_x:cur_x + h, cur_y:cur_y + w, 1] += dist[i, :, :]
    return occur_im

