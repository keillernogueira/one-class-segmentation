import gc
import sys
import os
import datetime
import imageio
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
import scipy.stats as stats

import torch
from torch import optim
from torch.autograd import Variable

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dataloaders.dataloader import DataLoader
from dataloaders.dataloader_orange import DataLoaderOrange
from dataloaders.dataloader_coffee import DataLoaderCoffee
from dataloaders.dataloader_coffee_full import DataLoaderCoffeeFull

from config import *
from utils import *
from network import FCNWideResNet50

from feat_ext import general_feature_extractor
from contrastive_loss import ContrastiveLoss
from contrastive_loss_double_margin import ContrastiveLossDoubleMargin
from learnt_prototypical import LearntPrototypes


def test_full_map(test_loader, criterion, net, epoch, output_path):
    # Setting network for evaluation mode.
    net.eval()

    prob_im = np.zeros([test_loader.dataset.labels[0].shape[0],
                        test_loader.dataset.labels[0].shape[1]], dtype=np.float64)
    occur_im = np.zeros([test_loader.dataset.labels[0].shape[0],
                         test_loader.dataset.labels[0].shape[1]], dtype=int)

    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs, cur_maps, cur_xs, cur_ys = data

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs = net(inps_c)

            for j in range(outs.shape[0]):
                # cur_map = cur_maps[j]
                cur_x = cur_xs[j]
                cur_y = cur_ys[j]

                outs_p = -outs.permute(0, 2, 3, 1).cpu().detach().numpy()

                prob_im[cur_x:cur_x + test_loader.dataset.crop_size,
                        cur_y:cur_y + test_loader.dataset.crop_size] += outs_p[j, :, :, 0]
                occur_im[cur_x:cur_x + test_loader.dataset.crop_size,
                         cur_y:cur_y + test_loader.dataset.crop_size] += 1

    if test_loader.dataset.dataset == 'Orange':
        pred_pos = np.where(occur_im.flatten() >= 1)

    # normalise to remove non-predicted pixels
    prob_im[np.where(occur_im == 0)] = 1
    occur_im[np.where(occur_im == 0)] = 1
    # prob_im_argmax = ((prob_im / occur_im.astype(float)) <= 0.999).astype(int)  # v1
    if hasattr(criterion, 'pos_margin'):
        prob_im_argmax = ((prob_im / occur_im.astype(float)) < criterion.pos_margin).astype(int)
    else:
        prob_im_argmax = ((prob_im / occur_im.astype(float)) < criterion.margin).astype(int)
    print(prob_im_argmax.shape, np.bincount(prob_im_argmax.flatten()),
          test_loader.dataset.labels[0].shape, np.bincount(test_loader.dataset.labels[0].flatten()))

    # pixel outside area should be 0 for the final image
    prob_im_argmax[np.where(test_loader.dataset.labels[0] == 2)] = 0
    # Saving predictions.
    imageio.imwrite(os.path.join(output_path, 'proto_prd.png'), prob_im_argmax*255)

    if test_loader.dataset.dataset == 'Coffee_Full':
        labs = test_loader.dataset.labels[0]
        coord = np.where(labs != 2)
        lbl = labs[coord]
        pred = prob_im_argmax[coord]
    elif test_loader.dataset.dataset == 'Orange':
        lbl = test_loader.dataset.labels[0].flatten()[pred_pos]
        pred = prob_im_argmax.flatten()[pred_pos]
    else:
        lbl = test_loader.dataset.labels[0].flatten()
        pred = prob_im_argmax.flatten()

    print(lbl.shape, np.bincount(lbl.flatten()), pred.shape, np.bincount(pred.flatten()))

    acc = accuracy_score(lbl, pred)
    conf_m = confusion_matrix(lbl, pred)
    f1_s_w = f1_score(lbl, pred, average='weighted')
    f1_s_micro = f1_score(lbl, pred, average='micro')
    f1_s_macro = f1_score(lbl, pred, average='macro')
    kappa = cohen_kappa_score(lbl, pred)
    tau, p = stats.kendalltau(lbl, pred)

    _sum = 0.0
    for k in range(len(conf_m)):
        _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

    print("---- Validation/Test -- Epoch " + str(epoch) +
          " -- Time " + str(datetime.datetime.now().time()) +
          " Overall Accuracy= " + "{:.4f}".format(acc) +
          " Normalized Accuracy= " + "{:.4f}".format(_sum / 2.0) +
          " F1 score weighted= " + "{:.4f}".format(f1_s_w) +
          " F1 score micro= " + "{:.4f}".format(f1_s_micro) +
          " F1 score macro= " + "{:.4f}".format(f1_s_macro) +
          " Kappa= " + "{:.4f}".format(kappa) +
          " Tau= " + "{:.4f}".format(tau) +
          " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
          )

    sys.stdout.flush()


def test(test_loader, criterion, net, epoch, margin):
    # Setting network for evaluation mode.
    net.eval()

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
            outs = net(inps_c)

            pred = (-outs < margin).int().detach().cpu().numpy().flatten()
            labs = labs.flatten()

            if test_loader.dataset.dataset == 'Coffee_Full':
                # filtering out pixels
                coord = np.where(labs != 2)
                labs = labs[coord]
                pred = pred[coord]

            track_cm += confusion_matrix(labs, pred, labels=[0, 1])

    acc = (track_cm[0][0] + track_cm[1][1])/np.sum(track_cm)
    f1_s = f1_with_cm(track_cm)

    _sum = 0.0
    for k in range(len(track_cm)):
        _sum += (track_cm[k][k] / float(np.sum(track_cm[k])) if np.sum(track_cm[k]) != 0 else 0)
    nacc = _sum / float(test_loader.dataset.num_classes)

    print("---- Validation/Test -- Epoch " + str(epoch) +
          " -- Time " + str(datetime.datetime.now().time()) +
          " -- Margin= " + str(margin) +
          " Overall Accuracy= " + "{:.4f}".format(acc) +
          " Normalized Accuracy= " + "{:.4f}".format(nacc) +
          " F1 Score= " + "{:.4f}".format(f1_s) +
          " Confusion Matrix= " + np.array_str(track_cm).replace("\n", "")
          )

    sys.stdout.flush()

    return acc, nacc


def train(train_loader, net, criterion, optimizer, epoch, output):
    # Setting network for training mode.
    net.train()

    # Average Meter for batch loss.
    train_loss = list()

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
        outs = net(inps)

        # computing loss
        loss = criterion(-outs, labs)
        # make_dot(loss).render("original", format="png")

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            # pred = (-outs <= 0.999).int().detach().cpu().numpy()  # v1
            if hasattr(criterion, 'pos_margin'):
                pred = (-outs < criterion.pos_margin).int().detach().cpu().numpy().flatten()
            else:
                pred = (-outs < criterion.margin).int().detach().cpu().numpy().flatten()
            labels = labels.flatten()

            if train_loader.dataset.dataset == 'Coffee_Full':
                # filtering out pixels
                coord = np.where(labels != 2)
                labels = labels[coord]
                pred = pred[coord]

            acc = accuracy_score(labels, pred)
            conf_m = confusion_matrix(labels, pred, labels=[0, 1])
            f1_s = f1_score(labels, pred, average='weighted')

            _sum = 0.0
            for k in range(len(conf_m)):
                _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i+1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / 2.0) +
                  " F1 Score= " + "{:.4f}".format(f1_s) +
                  " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )
            # project_data(torch.cat([criterion.protos[0][None, :], criterion.protos[1][None, :],
            #                         feat_flat[0:5000, :]]).cpu().detach().numpy(),
            #              torch.cat([torch.Tensor([2]).cuda(), torch.Tensor([3]).cuda(),
            #                         labs.view(-1)[0:5000]]).cpu().detach().numpy(),
            #              output + 'plot_' + str(epoch) + '_' + str(i) + '.png',
            #              pca_n_components=50)

    gc.collect()
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True, help='Operation to be performed]',
                        choices=['Train', 'Test', 'Plot', 'Test_Full'])
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save outcomes (such as images and trained models) of the algorithm.')

    # dataset options
    parser.add_argument('--dataset', type=str, required=True, help='Dataset.',
                        choices=['River', 'Orange', 'Coffee', 'Coffee_Full'])
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--training_images', type=str, nargs="+", required=False, help='Training image names.')
    parser.add_argument('--testing_images', type=str, nargs="+", required=False, help='Testing image names.')
    parser.add_argument('--crop_size', type=int, required=True, help='Crop size.')
    parser.add_argument('--stride_crop', type=int, required=True, help='Stride size')

    # model options
    parser.add_argument('--model', type=str, required=True, default=None,
                        help='Model to be used.', choices=['WideResNet'])
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model that can be load and used for inference.')
    parser.add_argument('--weights', type=float, nargs='+', default=[1.0, 1.0], help='Weight Loss.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=50, help='Number of epochs')

    # specific parameters
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for the contrastive learning')
    parser.add_argument('--pos_margin', type=float, default=None,
                        help='Margin for the positive class of the contrastive learning')
    parser.add_argument('--miner', type=str2bool, default=False,
                        help='Miner hard samples and equalize number fo samples 1:1')
    args = parser.parse_args()
    print(sys.argv[0], args)

    # data loaders
    if args.operation == 'Train':
        if args.dataset == 'River':
            print('---- training data ----')
            train_dataset = DataLoader('Train', args.dataset, args.dataset_path, args.training_images, args.crop_size,
                                       args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoader('Full_test', args.dataset, args.dataset_path, args.testing_images,
                                      args.crop_size, args.stride_crop, mean=train_dataset.mean, std=train_dataset.std)
        elif args.dataset == 'Orange':
            print('---- training data ----')
            train_dataset = DataLoaderOrange('Train', args.dataset, args.dataset_path, args.crop_size, args.stride_crop,
                                             output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderOrange('Test', args.dataset, args.dataset_path, args.crop_size, args.stride_crop,
                                            mean=train_dataset.mean, std=train_dataset.std)
        elif args.dataset == 'Coffee':
            print('---- training data ----')
            train_dataset = DataLoaderCoffee('Train', args.dataset, args.dataset_path, args.training_images,
                                             args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderCoffee('Test', args.dataset, args.dataset_path, args.testing_images,
                                            args.crop_size, args.stride_crop,
                                            mean=train_dataset.mean, std=train_dataset.std)
        elif args.dataset == 'Coffee_Full':
            print('---- training data ----')
            train_dataset = DataLoaderCoffeeFull('Train', args.dataset, args.dataset_path, args.training_images,
                                                 args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderCoffeeFull('Test', args.dataset, args.dataset_path, args.testing_images,
                                                args.crop_size, args.stride_crop,
                                                mean=train_dataset.mean, std=train_dataset.std)
        else:
            raise NotImplementedError("Dataset " + args.dataset + " not implemented")

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # network
        if args.model == 'WideResNet':
            model = LearntPrototypes(FCNWideResNet50(train_dataset.num_classes, pretrained=True, classif=False),
                                     squared=False, n_prototypes=1, embedding_dim=2560)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        # loss
        if args.pos_margin is not None:
            criterion = ContrastiveLossDoubleMargin(args.margin, args.pos_margin,
                                                    args.miner, args.weights, ignore_index=2)
        else:
            criterion = ContrastiveLoss(args.margin, args.miner, args.weights, ignore_index=2)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                               betas=(0.9, 0.99))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

        # load model and readjust scheduler
        curr_epoch = 1
        best_records = []
        if args.model_path is not None:
            print('Loading model ' + args.model_path)
            best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
            model.load_state_dict(torch.load(args.model_path))
            # optimizer.load_state_dict(torch.load(args.model_path.replace("model", "opt")))
            curr_epoch += int(os.path.basename(args.model_path)[:-4].split('_')[-1])
            for i in range(curr_epoch):
                scheduler.step()
        model.cuda()

        print('---- training ----')
        for epoch in range(curr_epoch, args.epoch_num + 1):
            train(train_dataloader, model, criterion, optimizer, epoch, args.output_path)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                for m in [0.5, 1.0, 1.5, 2.0]:
                    acc, nacc = test(test_dataloader, criterion, model, epoch, margin=m)
                    save_best_models(model, optimizer, args.output_path, best_records, epoch, nacc)
            scheduler.step()
    elif args.operation == 'Test':
        print('---- testing ----')
        best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
        index = 0
        for i in range(len(best_records)):
            if best_records[index]['nacc'] < best_records[i]['nacc']:
                index = i

        print('---- data ----')
        test_dataset = DataLoader('Validation', args.dataset_path, args.testing_images,
                                  args.crop_size, args.stride_crop, output_path=args.output_path)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # loss
        criterion = ContrastiveLoss(args.margin, args.miner, args.weights)

        # network
        if args.model == 'WideResNet':
            model = LearntPrototypes(FCNWideResNet50(test_dataloader.num_classes, pretrained=True, classif=False),
                                     squared=False, n_prototypes=1, embedding_dim=2560)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        epoch = int(best_records[index]['epoch'])
        print("loading model_" + str(epoch) + '.pth')
        model.load_state_dict(torch.load(os.path.join(args.output_path, 'model_' + str(epoch) + '.pth')))
        model.cuda()

        test(test_dataloader, criterion, model, epoch)
    elif args.operation == 'Test_Full':
        print('---- testing ----')

        best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
        index = 0
        for i in range(len(best_records)):
            if best_records[index]['nacc'] < best_records[i]['nacc']:
                index = i

        print('---- data ----')
        if args.dataset == 'River':
            print('---- testing data ----')
            test_dataset = DataLoader('Full_test', args.dataset, args.dataset_path, args.testing_images,
                                      args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Orange':
            print('---- testing data ----')
            test_dataset = DataLoaderOrange('Test', args.dataset, args.dataset_path, args.crop_size, args.stride_crop,
                                            output_path=args.output_path)
        elif args.dataset == 'Coffee':
            print('---- testing data ----')
            test_dataset = DataLoaderCoffee('Test', args.dataset, args.dataset_path, args.testing_images,
                                            args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Coffee_Full':
            print('---- testing data ----')
            test_dataset = DataLoaderCoffeeFull('Full_Test', args.dataset, args.dataset_path, args.testing_images,
                                                args.crop_size, args.stride_crop, output_path=args.output_path)
        else:
            raise NotImplementedError("Dataset " + args.dataset + " not implemented")

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # loss
        criterion = ContrastiveLoss(args.margin, args.miner, args.weights)

        # network
        if args.model == 'WideResNet':
            model = LearntPrototypes(FCNWideResNet50(test_dataset.num_classes, pretrained=True, classif=False),
                                     squared=False, n_prototypes=1, embedding_dim=2560)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        epoch = int(best_records[index]['epoch'])
        print("loading model_" + str(epoch) + '.pth')
        model.load_state_dict(torch.load(os.path.join(args.output_path, 'model_' + str(epoch) + '.pth')))
        model.cuda()

        test_full_map(test_dataloader, criterion, model, epoch, args.output_path)
    elif args.operation == 'Plot':
        print('---- plotting ----')
        best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
        index = 0
        for i in range(len(best_records)):
            if best_records[index]['nacc'] < best_records[i]['nacc']:
                index = i

        test_dataset = DataLoader('Plot', args.dataset_path, args.testing_images,
                                  args.crop_size, args.stride_crop, output_path=args.output_path)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # network
        if args.model == 'WideResNet':
            model = LearntPrototypes(FCNWideResNet50(test_dataset.num_classes, pretrained=True, classif=False),
                                     squared=False, n_prototypes=1, embedding_dim=2560)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        epoch = int(best_records[index]['epoch'])
        print("loading model_" + str(epoch) + '.pth')
        model.load_state_dict(torch.load(os.path.join(args.output_path, 'model_' + str(epoch) + '.pth')))
        model.cuda()

        print('---- extracting feat test ----')
        feats, lbs = general_feature_extractor(test_dataloader, model)
        lbs = lbs.reshape(-1)
        print('feats', feats.shape, lbs.shape, np.bincount(lbs[0:100000]))
        project_data(feats[0:100000, :], lbs[0:100000], args.output_path + 'plot.png', pca_n_components=50)
    else:
        raise NotImplementedError("Operation " + args.operation + " not implemented")
