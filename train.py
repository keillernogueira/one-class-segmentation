import gc
import sys
import datetime

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from torchviz import make_dot
from utils import *
from config import *
from triplet_losses import batch_hard_triplet_loss, batch_all_triplet_loss
from prototypical import prototypical_loss


def train(train_loader, net, tl_criterion, optimizer, epoch, alpha, margin,
          train_strategy, test_strategy, ce_criterion=None, output=None):
    # Setting network for training mode.
    net.train()

    # Average Meter for batch loss.
    train_loss = list()

    track_mean = None
    track_mean_return = None
    acc = 0.0

    # Iterating over batches.
    for i, data in enumerate(train_loader):
        # Obtaining buzz sounds and labels
        inps, labels = data[0], data[1]

        # Casting to cuda variables.
        inps = Variable(inps).cuda()
        labs = Variable(labels).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        _, fv2, fv4 = net(inps)

        # if outs is not None:
        #     soft_outs = F.softmax(outs, dim=1)
        #     # Obtaining predictions.
        #     prds = soft_outs.cpu().data.numpy().argmax(axis=1)

        # Computing Cross entropy loss.
        # loss_ce = ce_criterion(outs, labs)

        # computing triplet loss
        feat_flat = torch.cat([fv2, fv4], 1)
        feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(-1, feat_flat.size(1))

        if train_strategy == 'original':
            # main, knn
            a, p, n, aux = get_triples(feat_flat, labs.view(-1), track_mean)
        elif train_strategy == 'detach':
            # v2
            a, p, n, aux = get_triples_detach(feat_flat, labs.view(-1), track_mean)
        elif train_strategy == 'track_mean':
            # track mean
            a, p, n, track_mean = get_triples_track(feat_flat, labs.view(-1), track_mean, alpha)
        elif train_strategy == 'hard_triplet':
            # hard triples
            a, p, n, aux = get_hard_triples(feat_flat, labs.view(-1), track_mean)

        if train_strategy == 'std_hard_triplet':
            # standard
            loss = batch_hard_triplet_loss(labs.view(-1), feat_flat, margin=margin)
        # elif train_strategy == 'prototypical':
        #     loss, acc, _ = prototypical_loss(feat_flat, labs.view(-1))
        else:
            loss = tl_criterion(a, p, n)
        # make_dot(loss).render("original", format="png")

        # loss = loss_ce + loss_tl

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        if test_strategy == 'track_mean' and train_strategy != 'track_mean':
            if track_mean_return is None:
                track_mean_return = aux.detach().cpu()
            else:
                new = torch.mean(torch.stack((track_mean_return, aux.detach().cpu()), dim=0), dim=0)
                # diff = torch.cdist(torch.unsqueeze(track_mean_return, dim=0), torch.unsqueeze(new, dim=0), p=2).item()
                track_mean_return = new

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            if train_strategy != 'std_hard_triplet':
                acc = calc_accuracy_triples(a, p, n, margin=margin)
            #     acc = accuracy_score(labels.flatten(), prds.flatten())
            #     conf_m = confusion_matrix(labels.flatten(), prds.flatten())
            #
            #     _sum = 0.0
            #     for k in range(len(conf_m)):
            #         _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)
            #
            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i+1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc))
        #           " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
        #           " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
        #           )
        #     project_data(feat_flat[0:5000, :].cpu().detach().numpy(),
        #                  labs.view(-1)[0:5000].cpu().detach().numpy(),
        #                  output + 'plot_' + str(epoch) + '_' + str(i) + '.png',
        #                  pca_n_components=50)

    gc.collect()
    sys.stdout.flush()

    if test_strategy == 'track_mean' and train_strategy != 'track_mean':
        return track_mean_return
    elif test_strategy == 'track_mean' and train_strategy == 'track_mean':
        return track_mean
    else:
        return None


# backup
def original_train(train_loader, net, ce_criterion, tl_criterion, optimizer, epoch, output):
    # Setting network for training mode.
    net.train()

    # Average Meter for batch loss.
    train_loss = list()

    track_mean = None
    track_mean_return = None

    # Iterating over batches.
    for i, data in enumerate(train_loader):
        # Obtaining buzz sounds and labels
        inps, labels = data[0], data[1]

        # Casting to cuda variables.
        inps = Variable(inps).cuda()
        labs = Variable(labels).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs, fv2, fv4 = net(inps)

        if outs is not None:
            soft_outs = F.softmax(outs, dim=1)
            # Obtaining predictions.
            prds = soft_outs.cpu().data.numpy().argmax(axis=1)

        # Computing Cross entropy loss.
        # loss_ce = ce_criterion(outs, labs)

        # computing triplet loss
        feat_flat = torch.cat([fv2, fv4], 1)
        feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(-1, feat_flat.size(1))
        a, p, n, aux = get_triples(feat_flat, labs.view(-1), track_mean)
        loss = tl_criterion(a, p, n)
        # make_dot(loss).render("original", format="png")

        # loss = loss_ce + loss_tl

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())
        if track_mean_return is None:
            track_mean_return = aux.detach().cpu()
        else:
            new = torch.mean(torch.stack((track_mean_return, aux.detach().cpu()), dim=0), dim=0)
            diff = torch.cdist(torch.unsqueeze(track_mean_return, dim=0), torch.unsqueeze(new, dim=0), p=2).item()
            track_mean_return = new

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            acc = calc_accuracy_triples(a, p, n)
            #     acc = accuracy_score(labels.flatten(), prds.flatten())
            #     conf_m = confusion_matrix(labels.flatten(), prds.flatten())
            #
            #     _sum = 0.0
            #     for k in range(len(conf_m)):
            #         _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)
            #
            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i+1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Mean Diff " + "{:.4f}".format(diff))
        #           " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
        #           " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
        #           )
        #     project_data(feat_flat[0:5000, :].cpu().detach().numpy(),
        #                  labs.view(-1)[0:5000].cpu().detach().numpy(),
        #                  output + 'plot_' + str(epoch) + '_' + str(i) + '.png',
        #                  pca_n_components=50)

    gc.collect()
    sys.stdout.flush()

    return track_mean_return
