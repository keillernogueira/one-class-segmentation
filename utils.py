import os
import argparse
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import imageio

from sklearn import decomposition
from sklearn.manifold import TSNE
# from tsne_torch import TorchTSNE as TSNE

import torch
from torch import nn
from torch.autograd import Variable

from config import NUM_WORKERS


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


def save_best_models(net, optimizer, output_path, best_records, epoch, metric, num_saves=3, track_mean=None):
    print('check', math.isnan(metric))
    if math.isnan(metric):
        metric = 0.0
    print('after metric', metric)
    if len(best_records) < num_saves:
        best_records.append({'epoch': epoch, 'kappa': metric, 'track_mean': track_mean})

        torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
        # torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
    else:
        # find min saved acc
        min_index = 0
        for i, r in enumerate(best_records):
            if best_records[min_index]['kappa'] > best_records[i]['kappa']:
                min_index = i
        print('before', best_records, best_records[min_index], metric,
              best_records[min_index]['kappa'], metric > best_records[min_index]['kappa'])
        # check if currect acc is greater than min saved acc
        if metric > best_records[min_index]['kappa']:
            # if it is, delete previous files
            min_step = str(best_records[min_index]['epoch'])

            os.remove(os.path.join(output_path, 'model_' + min_step + '.pth'))
            # os.remove(os.path.join(output_path, 'opt_' + min_step + '.pth'))

            # replace min value with current
            best_records[min_index] = {'epoch': epoch, 'kappa': metric, 'track_mean': track_mean}

            # save current model
            torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
            # torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
    print('after', best_records)
    np.save(os.path.join(output_path, 'best_records.npy'), best_records)


def project_data(prototype, train_data, train_labels, test_data, test_labels, save_name,
                 num_samples=1000, pca_n_components=50):
    # distance between all embeddings and the prototype
    train_dists = np.linalg.norm(train_data[:, np.newaxis, :] - prototype[np.newaxis, :, :], axis=-1)
    test_dists = np.linalg.norm(test_data[:, np.newaxis, :] - prototype[np.newaxis, :, :], axis=-1)

    train_pos_data = train_data[np.where(train_labels == 1)]
    train_pos_dist = np.squeeze(train_dists[np.where(train_labels == 1)])
    train_neg_data = train_data[np.where(train_labels == 0)]
    train_neg_dist = np.squeeze(train_dists[np.where(train_labels == 0)])

    test_pos_data = test_data[np.where(test_labels == 1)]
    test_pos_dist = np.squeeze(test_dists[np.where(test_labels == 1)])
    test_neg_data = test_data[np.where(test_labels == 0)]
    test_neg_dist = np.squeeze(test_dists[np.where(test_labels == 0)])

    print(train_dists.shape, test_dists.shape,
          train_pos_data.shape, train_neg_data.shape, train_pos_dist.shape, train_neg_dist.shape,
          test_pos_data.shape, test_neg_data.shape, test_pos_dist.shape, test_neg_dist.shape,)

    train_pos_zip = zip(train_pos_data, train_pos_dist)
    train_neg_zip = zip(train_neg_data, train_neg_dist)
    test_pos_zip = zip(test_pos_data, test_pos_dist)
    test_neg_zip = zip(test_neg_data, test_neg_dist)

    train_pos_sorted = list(sorted(train_pos_zip, key=lambda x: x[1]))
    train_neg_sorted = list(sorted(train_neg_zip, key=lambda x: x[1]))
    test_pos_sorted = list(sorted(test_pos_zip, key=lambda x: x[1]))
    test_neg_sorted = list(sorted(test_neg_zip, key=lambda x: x[1]))

    # train_neg_sorted = list(train_neg_sorted)
    # print(np.asarray(train_neg_sorted).shape)
    # print(train_neg_sorted[0][1], train_neg_sorted[1][1], train_neg_sorted[2][1])
    # print(train_neg_sorted[-3][1], train_neg_sorted[-2][1], train_neg_sorted[-1][1])

    selected_train_pos = train_pos_sorted[:min(num_samples, len(train_pos_sorted))]
    selected_train_neg = train_neg_sorted[-min(num_samples, len(train_neg_sorted)):]
    selected_test_pos = test_pos_sorted[:min(num_samples, len(test_pos_sorted))]
    selected_test_neg = test_neg_sorted[-min(num_samples, len(test_neg_sorted)):]

    train_pos, train_pos_dist = map(list, zip(*selected_train_pos))
    train_neg, train_neg_dist = map(list, zip(*selected_train_neg))
    test_pos, test_pos_dist = map(list, zip(*selected_test_pos))
    test_neg, test_neg_dist = map(list, zip(*selected_test_neg))

    print(np.asarray(train_pos).shape, np.asarray(train_pos_dist).shape, train_pos_dist[0:3],
          np.asarray(train_neg).shape, np.asarray(train_neg_dist).shape, train_neg_dist[0:3],
          np.asarray(test_pos).shape, np.asarray(test_pos_dist).shape, test_pos_dist[0:3],
          np.asarray(test_neg).shape, np.asarray(test_neg_dist).shape, test_neg_dist[0:3],
          prototype.shape)

    data = np.concatenate((train_pos, train_neg, test_pos, test_neg, prototype))
    labels = np.concatenate((np.ones(len(train_neg), dtype=int),
                             np.full(len(test_neg), 2, dtype=int),
                             np.full(len(train_pos), 3, dtype=int),
                             np.full(len(test_pos), 4, dtype=int),
                             np.asarray([0])))

    print('final', data.shape, labels.shape, np.bincount(labels))

    # pca_model = decomposition.PCA(n_components=pca_n_components, random_state=12345)
    # pca_data = pca_model.fit_transform(data)
    # tsne_model = TSNE(n_components=2, n_jobs=-1, learning_rate='auto', init='random')  # original
    tsne_model = TSNE(n_components=2, perplexity=50, early_exaggeration=70, n_jobs=-1,
                      init='pca', learning_rate=50, n_iter=5000)
    tsne_data = tsne_model.fit_transform(data)
    print('tsne_data', tsne_data.shape)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=labels,
                    palette=sns.color_palette("hls", 5), legend="full", alpha=0.3)
    # plt.show()
    plt.savefig(save_name)


def sample_weight_train_loader(train_dataset, gen_classes, batch_size):
    class_loader_weights = 1. / np.bincount(gen_classes)
    samples_weights = class_loader_weights[gen_classes]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   num_workers=NUM_WORKERS, drop_last=False, sampler=sampler)
    return train_dataloader


def kappa_with_cm(conf_matrix):
    acc = 0
    marginal = 0
    total = float(np.sum(conf_matrix))
    for i in range(len(conf_matrix)):
        acc += conf_matrix[i][i]
        marginal += np.sum(conf_matrix, 0)[i] * np.sum(conf_matrix, 1)[i]

    kappa = (total * acc - marginal) / (total * total - marginal)
    return kappa


def f1_with_cm(conf_matrix):
    precision = [0] * len(conf_matrix)
    recall = [0] * len(conf_matrix)
    f1 = [0] * len(conf_matrix)
    for i in range(len(conf_matrix)):
        precision[i] = conf_matrix[i][i] / float(np.sum(conf_matrix, 0)[i])
        recall[i] = conf_matrix[i][i] / float(np.sum(conf_matrix, 1)[i])
        f1[i] = 2 * ((precision[i]*recall[i])/(precision[i]+recall[i]))

    return np.mean(f1)


def jaccard_with_cm(conf_matrix):
    den = float(np.sum(conf_matrix[:, 1]) + np.sum(conf_matrix[1]) - conf_matrix[1][1])
    _sum_iou = conf_matrix[1][1] / den if den != 0 else 0

    return _sum_iou


def get_triples(feat, labs, track_mean=None):
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
           torch.index_select(feat[labs == 0, :], 0, topk_neg), \
           track_mean


def get_triples_detach(feat_g, labs, track_mean):
    feat = feat_g.clone().detach()

    # detach
    feat_pos_lbl_d = feat[labs == 1, :]

    # attached
    feat_pos_lbl = feat_g[labs == 1, :]
    mean_pos_lbl = torch.mean(feat_pos_lbl, dim=0)
    if track_mean is None:
        track_mean = mean_pos_lbl
    else:
        track_mean = torch.mean(torch.stack((track_mean, mean_pos_lbl), dim=0), dim=0)
    track_mean_d = track_mean.clone().detach()

    cdist_neg_lbl = torch.cdist(torch.unsqueeze(track_mean_d, dim=0), feat[labs == 0, :], p=2).squeeze()
    _, topk_neg = torch.topk(cdist_neg_lbl, torch.count_nonzero(labs), largest=False)

    return torch.unsqueeze(track_mean, dim=0).repeat(feat_pos_lbl_d.size(0), 1), \
           feat_pos_lbl_d, \
           torch.index_select(feat[labs == 0, :], 0, topk_neg), \
           track_mean


def get_triples_track(feat, labs, track_mean, alpha=0.6):
    feat_pos_lbl = feat[labs == 1, :]
    mean_pos_lbl = torch.mean(feat_pos_lbl, dim=0)
    if track_mean is None:
        track_mean = mean_pos_lbl
    else:
        track_mean = torch.mean(torch.stack((track_mean, mean_pos_lbl), dim=0), dim=0)
        # track_mean = alpha * track_mean + (1. - alpha) * mean_pos_lbl

    cdist_neg_lbl = torch.cdist(torch.unsqueeze(track_mean, dim=0), feat[labs == 0, :], p=2).squeeze()
    _, topk_neg = torch.topk(cdist_neg_lbl, torch.count_nonzero(labs), largest=False)

    return torch.unsqueeze(track_mean, dim=0).repeat(feat_pos_lbl.size(0), 1), \
           feat_pos_lbl, \
           torch.index_select(feat[labs == 0, :], 0, topk_neg), \
           track_mean.clone().detach()


def get_hard_triples(feat, labs, track_mean, margin=1):
    # calculating centroid value
    feat_pos_lbl = feat[labs == 1, :]
    neg_pos_lbl = feat[labs == 0, :]

    mean_pos_lbl = torch.mean(feat_pos_lbl, dim=0)
    if track_mean is None:
        track_mean = mean_pos_lbl
    else:
        track_mean = torch.mean(torch.stack((track_mean, mean_pos_lbl), dim=0), dim=0)

    # dist centroid to POS samples
    cdist_pos, cdist_pos_idx = torch.cdist(torch.unsqueeze(track_mean, dim=0),
                                           feat_pos_lbl, p=2).squeeze().sort(0, descending=True)

    # dist centroid to NEG samples
    cdist_neg, cdist_neg_idx = torch.cdist(torch.unsqueeze(track_mean, dim=0),
                                           neg_pos_lbl, p=2).squeeze().sort(0)

    hard = cdist_pos >= (cdist_neg[0:cdist_pos.shape[0]] - margin)

    return torch.unsqueeze(track_mean, dim=0).repeat(torch.count_nonzero(hard), 1), \
           feat_pos_lbl[cdist_pos_idx[hard]], \
           neg_pos_lbl[cdist_neg_idx[0:cdist_pos.shape[0]][hard]], \
           track_mean


def calc_accuracy_triples(a, p, n, margin=1):
    a_p_dist = torch.cdist(a[0:1, :], p, p=2)
    a_n_dist = torch.cdist(a[0:1, :], n, p=2)
    dist = a_n_dist - margin > a_p_dist

    return torch.count_nonzero(dist) / a.size(0)


def predict_patches(feat_flat, track_mean):
    # m, h, w, f = feat_flat.shape
    # feat_flat = feat_flat.view(-1, feat_flat.size(3))
    dist = torch.cdist(torch.unsqueeze(track_mean, dim=0), feat_flat, p=2).squeeze()
    pred = (dist < 1).int()  # .view(m, h, w)
    # print('print', pred.shape, torch.bincount(pred.view(-1)))
    return pred


# https://www.kaggle.com/ashishpatel26/triplet-loss-network-for-humpback-whale-prediction
def predict_map(occur_im, feat_flat, track_mean, maps, curxs, curys):
    m, h, w, f = feat_flat.shape
    pred = predict_patches(feat_flat, track_mean)
    for i in range(len(maps)):
        cur_x = curxs[i]
        cur_y = curys[i]

        occur_im[cur_x:cur_x + h, cur_y:cur_y + w, 0] += (1-pred[i, :, :])
        occur_im[cur_x:cur_x + h, cur_y:cur_y + w, 1] += pred[i, :, :]
    return occur_im


def calculate_mask_distribution(dataset_path, images):
    overall_distr = np.zeros(2)
    for img in images:
        temp_mask = imageio.imread(os.path.join(dataset_path, img + '.tif')).astype(int)
        print(temp_mask.shape)
        bin = np.bincount(temp_mask.flatten(), minlength=2)
        print("bin ", img, bin)
        overall_distr += bin
    print("overall distr", overall_distr)
