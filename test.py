import sys
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from torch.autograd import Variable

from clustering import KNN
from utils import *
from prototypical import prototypical_loss, ProtoPred


def test_full_map(test_loader, net, epoch, track_mean=None):
    # Setting network for evaluation mode.
    net.eval()

    occur_im = torch.zeros([test_loader.dataset.labels[0].shape[0],
                            test_loader.dataset.labels[0].shape[1], test_loader.dataset.num_classes],
                           dtype=torch.int)
    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs, maps, curxs, curys = data

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs, fv2, fv4 = net(inps_c)
            # Computing probabilities.
            # soft_outs = F.softmax(outs, dim=1)

            # Obtaining prior predictions.
            # prds = soft_outs.cpu().data.numpy().argmax(axis=1)

            feat_flat = torch.cat([fv2, fv4], 1)
            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().detach().cpu()
            if track_mean is not None:
                occur_im = predict_map(occur_im, feat_flat, track_mean, maps, curxs, curys)

    if track_mean is not None:
        pred = torch.argmax(occur_im, dim=-1).contiguous().detach().cpu().numpy()
        acc = accuracy_score(test_loader.dataset.labels[0].flatten(), pred.flatten())
        conf_m = confusion_matrix(test_loader.dataset.labels[0].flatten(), pred.flatten())

        _sum = 0.0
        for k in range(len(conf_m)):
            _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
              " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
              )

        sys.stdout.flush()

    return 0, 0


def test_per_patch(test_loader, net, epoch, test_strategy, track_mean=None, val_dataloader=None, k=None):
    # Setting network for evaluation mode.
    net.eval()

    track_cm = np.zeros((2, 2))

    selected_val_samples = None
    selected_cal_labs = None
    knn_model = None
    proto_model = None

    if val_dataloader is not None:
        with torch.no_grad():
            # extracting features for KNN
            for i, data in enumerate(val_dataloader):
                # Obtaining images, labels and paths for batch.
                inps, labs = data[0], data[1]

                # Casting to cuda variables.
                inps_c = Variable(inps).cuda()

                # Forwarding.
                _, fv2, fv4 = net(inps_c)

                feat_flat = torch.cat([fv2, fv4], 1)
                feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(-1, feat_flat.size(1)).detach().cpu()
                labs = labs.view(-1).cpu()

                selected_val_samples = feat_flat
                selected_cal_labs = labs

            # print("Val data", selected_val_samples.shape, selected_cal_labs.shape, torch.bincount(selected_cal_labs))
            # if test_strategy == 'knn':
            # pca_model = decomposition.PCA(n_components=50, random_state=12345)
            # pca_data = pca_model.fit_transform(selected_val_samples)
            knn_model = KNN(selected_val_samples, selected_cal_labs, k=k)
            # else:
            #     proto_model = ProtoPred(selected_val_samples, selected_cal_labs)

    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs = data[0], data[1]

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            _, fv2, fv4 = net(inps_c)

            feat_flat = torch.cat([fv2, fv4], 1)
            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(-1, feat_flat.size(1)).detach().cpu()
            # labs = labs.detach().cpu().numpy()

            if test_strategy == 'track_mean' and track_mean is not None:
                preds = predict_patches(feat_flat, track_mean).detach().cpu().numpy()
            elif test_strategy == 'knn':
                preds = knn_model.predict(feat_flat)
            # else:
            #     preds = proto_model.predict(feat_flat)
            track_cm += confusion_matrix(labs.flatten(), preds.flatten(), labels=[0, 1])

    acc = (track_cm[0][0] + track_cm[1][1])/np.sum(track_cm)

    _sum = 0.0
    for k in range(len(track_cm)):
        _sum += (track_cm[k][k] / float(np.sum(track_cm[k])) if np.sum(track_cm[k]) != 0 else 0)
    nacc = _sum / float(test_loader.dataset.num_classes)

    print("---- Validation/Test -- Epoch " + str(epoch) +
          " -- Time " + str(datetime.datetime.now().time()) +
          " Overall Accuracy= " + "{:.4f}".format(acc) +
          " Normalized Accuracy= " + "{:.4f}".format(nacc) +
          " Confusion Matrix= " + np.array_str(track_cm).replace("\n", "")
          )

    sys.stdout.flush()

    return acc, nacc


# backup
def test_per_patch_o(test_loader, net, epoch, track_mean=None):
    # Setting network for evaluation mode.
    net.eval()

    acc = 0.0
    nacc = 0.0
    track_cm = np.zeros((2, 2))
    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs = data[0], data[1]

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs, fv2, fv4 = net(inps_c)
            # Computing probabilities.
            # soft_outs = F.softmax(outs, dim=1)

            # Obtaining prior predictions.
            # prds = soft_outs.cpu().data.numpy().argmax(axis=1)

            feat_flat = torch.cat([fv2, fv4], 1)
            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().detach().cpu()
            preds = predict_patches(feat_flat, track_mean).detach().cpu().numpy()
            labs = labs.detach().cpu().numpy()
            track_cm += confusion_matrix(labs.flatten(), preds.flatten(), labels=[0, 1])

    if track_mean is not None:
        acc = (track_cm[0][0] + track_cm[1][1])/np.sum(track_cm)

        _sum = 0.0
        for k in range(len(track_cm)):
            _sum += (track_cm[k][k] / float(np.sum(track_cm[k])) if np.sum(track_cm[k]) != 0 else 0)
        nacc = _sum / float(test_loader.dataset.num_classes)

        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(nacc) +
              " Confusion Matrix= " + np.array_str(track_cm).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, nacc


def test_per_patch_knn(knn_dataloader, test_loader, net, epoch, k=10):
    # Setting network for evaluation mode.
    net.eval()

    track_cm = np.zeros((2, 2))

    selected_train_samples_for_knn = None
    selected_train_labs_for_knn = None

    with torch.no_grad():
        # extracting features for KNN
        for i, data in enumerate(knn_dataloader):
            # Obtaining images, labels and paths for batch.
            inps, labs = data[0], data[1]

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()

            # Forwarding.
            _, fv2, fv4 = net(inps_c)

            feat_flat = torch.cat([fv2, fv4], 1)
            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(-1, feat_flat.size(1)).detach().cpu()
            labs = labs.view(-1).cpu()

            selected_train_samples_for_knn = feat_flat
            selected_train_labs_for_knn = labs

        # pca_model = decomposition.PCA(n_components=50, random_state=12345)
        # pca_data = pca_model.fit_transform(selected_train_samples_for_knn)
        knn_model = KNN(selected_train_samples_for_knn, selected_train_labs_for_knn, k=k)

        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs = data[0], data[1]

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs, fv2, fv4 = net(inps_c)

            feat_flat = torch.cat([fv2, fv4], 1)
            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(-1, feat_flat.size(1)).detach().cpu()

            # pca_data = pca_model.transform(feat_flat)
            preds = knn_model.predict(feat_flat)
            track_cm += confusion_matrix(labs.flatten(), preds.flatten(), labels=[0, 1])

    acc = (track_cm[0][0] + track_cm[1][1])/np.sum(track_cm)

    _sum = 0.0
    for k in range(len(track_cm)):
        _sum += (track_cm[k][k] / float(np.sum(track_cm[k])) if np.sum(track_cm[k]) != 0 else 0)
    nacc = _sum / float(test_loader.dataset.num_classes)

    print("---- Validation/Test -- Epoch " + str(epoch) +
          " -- Time " + str(datetime.datetime.now().time()) +
          " Overall Accuracy= " + "{:.4f}".format(acc) +
          " Normalized Accuracy= " + "{:.4f}".format(nacc) +
          " Confusion Matrix= " + np.array_str(track_cm).replace("\n", "")
          )

    sys.stdout.flush()

    return acc, nacc
