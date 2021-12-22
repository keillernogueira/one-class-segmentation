import gc
import sys
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchviz import make_dot

from dataloader import DataLoader
from config import *
from utils import *
from network import FCNWideResNet50


def test_full(test_loader, net, epoch, track_mean=None):
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


def test_per_patch(test_loader, net, epoch, track_mean=None):
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


def plot(test_loader, net):
    # Setting network for evaluation mode.
    net.eval()

    all_labels = None
    all_feats = None
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
            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(-1, feat_flat.size(1)).cpu().detach().numpy()

            if all_labels is None:
                all_labels = labs
                all_feats = feat_flat
            else:
                all_labels = np.concatenate((all_labels, labs))
                all_feats = np.concatenate((all_feats, feat_flat))

    return all_feats, all_labels


def train(train_loader, net, ce_criterion, tl_criterion, optimizer, epoch, output):
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
        a, p, n, aux = get_hard_triples(feat_flat, labs.view(-1), track_mean)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True, help='Operation to be performed]',
                        choices=['Train', 'Test', 'Plot'])
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save outcomes (such as images and trained models) of the algorithm.')

    # dataset options
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--training_images', type=str, nargs="+", required=True, help='Training image names.')
    parser.add_argument('--testing_images', type=str, nargs="+", required=True, help='Testing image names.')
    parser.add_argument('--crop_size', type=int, required=True, help='Crop size.')
    parser.add_argument('--stride_crop', type=int, required=True, help='Stride size')

    # model options
    parser.add_argument('--model', type=str, required=True, default=None,
                        help='Model to be used.', choices=['WideResNet'])
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model that can be load and used for inference.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()
    print(args)

    # data loaders
    if args.operation == 'Train':
        print('---- training data ----')
        train_dataset = DataLoader('Train', args.dataset_path, args.training_images, args.crop_size,
                                   args.stride_crop, output_path=args.output_path)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
        print('---- testing data ----')
        test_dataset = DataLoader('Validation', args.dataset_path, args.testing_images,
                                  args.crop_size, args.stride_crop, mean=train_dataset.mean, std=train_dataset.std)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # network
        if args.model == 'WideResNet':
            model = FCNWideResNet50(train_dataset.num_classes, pretrained=True, classif=False)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")
        model.cuda()

        # loss
        ce_criterion = nn.CrossEntropyLoss().cuda()
        tl_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                               betas=(0.9, 0.99))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

        curr_epoch = 1
        best_records = []
        print('---- training ----')
        for epoch in range(curr_epoch, args.epoch_num + 1):
            track_mean = train(train_dataloader, model, ce_criterion, tl_criterion,
                               optimizer, epoch, args.output_path)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                acc, nacc = test_per_patch(test_dataloader, model, epoch, track_mean)
                save_best_models(model, optimizer, args.output_path, best_records, epoch, nacc)

            scheduler.step()
    elif args.operation == 'Test':
        print('---- testing ----')
        # assert args.model_path is not None, "For inference, flag --model_path should be set."

        test_dataset = DataLoader('Validation', args.dataset_path, args.testing_images,
                                  args.crop_size, args.stride_crop, output_path=args.output_path)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # network
        if args.model == 'WideResNet':
            model = FCNWideResNet50(test_dataset.num_classes, pretrained=True, classif=False)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        epoch = 0
        if args.model_path is not None:
            model.load_state_dict(torch.load(args.model_path))
            epoch = int(os.path.basename(args.model_path)[:-4].split('_')[-1])
        model.cuda()

        test_per_patch(test_dataloader, model, epoch)
    elif args.operation == 'Plot':
        print('---- plotting ----')
        test_dataset = DataLoader('Plot', args.dataset_path, args.testing_images,
                                  args.crop_size, args.stride_crop, output_path=args.output_path)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # network
        if args.model == 'WideResNet':
            model = FCNWideResNet50(test_dataset.num_classes, pretrained=True, classif=False)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        epoch = 0
        if args.model_path is not None:
            model.load_state_dict(torch.load(args.model_path))
            epoch = int(os.path.basename(args.model_path)[:-4].split('_')[-1])
        model.cuda()

        feats, lbs = plot(test_dataloader, model)
        lbs = lbs.reshape(-1)
        print('feats', feats.shape, lbs.shape, np.bincount(lbs[0:100000]))
        project_data(feats[0:100000, :], lbs[0:100000], args.output_path + 'plot.png', pca_n_components=50)
    else:
        raise NotImplementedError("Operation " + args.operation + " not implemented")
