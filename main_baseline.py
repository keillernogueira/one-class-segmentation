import sys
import datetime
import imageio
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, \
    jaccard_score, precision_score, recall_score
import scipy.stats as stats

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dataloaders.dataloader import DataLoader
from dataloaders.dataloader_road import DataLoaderRoad
from dataloaders.dataloader_orange import DataLoaderOrange
from dataloaders.dataloader_coffee import DataLoaderCoffee
from dataloaders.dataloader_coffee_full import DataLoaderCoffeeFull
from dataloaders.dataloader_coffee_crop import DataLoaderCoffeeCrop
from dataloaders.dataloader_tree import DataLoaderTree

from config import *
from utils import *
from networks.FCNWideResNet50 import FCNWideResNet50
from networks.efficientnet import FCNEfficientNetB0
from networks.FCNDenseNet121 import FCNDenseNet121
from focal_loss import BinaryFocalLoss, FocalLossV2
from unified_focal_loss import UnifiedFocalLoss
from dual_focal_loss import DualFocalLoss


def test_full_map_one_map(test_loader, net, epoch, output_path):
    # Setting network for evaluation mode.
    net.eval()

    prob_im = np.zeros([test_loader.dataset.labels[0].shape[0],
                        test_loader.dataset.labels[0].shape[1], test_loader.dataset.num_classes], dtype=np.float64)
    occur_im = np.zeros([test_loader.dataset.labels[0].shape[0],
                         test_loader.dataset.labels[0].shape[1], test_loader.dataset.num_classes], dtype=int)

    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs, cur_maps, cur_xs, cur_ys = data

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs, _ = net(inps_c)
            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)

            for j in range(soft_outs.shape[0]):
                # cur_map = cur_maps[j]
                cur_x = cur_xs[j]
                cur_y = cur_ys[j]

                soft_outs_p = soft_outs.permute(0, 2, 3, 1).cpu().detach().numpy()

                prob_im[cur_x:cur_x + test_loader.dataset.crop_size,
                        cur_y:cur_y + test_loader.dataset.crop_size, :] += soft_outs_p[j, :, :, :]
                occur_im[cur_x:cur_x + test_loader.dataset.crop_size,
                         cur_y:cur_y + test_loader.dataset.crop_size, :] += 1

    occur_im[np.where(occur_im == 0)] = 1
    prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=-1)
    print('test_full check', prob_im_argmax.shape, np.bincount(prob_im_argmax.flatten()),
          test_loader.dataset.labels[0].shape, np.bincount(test_loader.dataset.labels[0].flatten()))

    # pixel outside area should be 0 for the final image
    prob_im_argmax[np.where(test_loader.dataset.labels[0] == 2)] = 0
    # Saving predictions.
    imageio.imwrite(os.path.join(output_path, 'baseline_prd.png'), prob_im_argmax*255)

    # filtering out pixels
    labs = test_loader.dataset.labels[0]
    coord = np.where(labs != 2)
    labs = labs[coord]
    prds = prob_im_argmax[coord]
    print(labs.shape, prds.shape, np.bincount(labs.flatten()), np.bincount(prds.flatten()))

    acc = accuracy_score(labs, prds)
    conf_m = confusion_matrix(labs, prds)
    f1_s_w = f1_score(labs, prds, average='weighted')
    f1_s_micro = f1_score(labs, prds, average='micro')
    f1_s_macro = f1_score(labs, prds, average='macro')
    kappa = cohen_kappa_score(labs, prds)
    jaccard = jaccard_score(labs, prds)
    tau, p = stats.kendalltau(labs, prds)

    _sum = 0.0
    for k in range(len(conf_m)):
        _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

    print("---- Validation/Test -- Epoch " + str(epoch) +
          " -- Time " + str(datetime.datetime.now().time()) +
          " Overall Accuracy= " + "{:.4f}".format(acc) +
          " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
          " F1 score weighted= " + "{:.4f}".format(f1_s_w) +
          " F1 score micro= " + "{:.4f}".format(f1_s_micro) +
          " F1 score macro= " + "{:.4f}".format(f1_s_macro) +
          " Kappa= " + "{:.4f}".format(kappa) +
          " Jaccard= " + "{:.4f}".format(jaccard) +
          " Tau= " + "{:.4f}".format(tau) +
          " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
          )

    sys.stdout.flush()


def test_full_map(test_loader, net, epoch, output_path):
    # Setting network for evaluation mode.
    net.eval()

    average_acc = 0.0
    average_n_acc = 0.0
    average_conf_m = np.zeros((2, 2))
    average_f1_s_w = 0.0
    average_f1_s_micro = 0.0
    average_f1_s_macro = 0.0
    average_kappa = 0.0
    average_jaccard = 0.0
    average_tau = 0.0
    average_prec_micro = 0.0
    average_prec_macro = 0.0
    average_prec_binary = 0.0
    average_rec_micro = 0.0
    average_rec_macro = 0.0
    average_rec_binary = 0.0

    prob_im = []
    occur_im = []
    for i in range(len(test_loader.dataset.labels)):
        prob_im.append(np.zeros([test_loader.dataset.labels[i].shape[0], test_loader.dataset.labels[i].shape[1],
                                 test_loader.dataset.num_classes], dtype=np.float32))
        occur_im.append(np.zeros([test_loader.dataset.labels[i].shape[0], test_loader.dataset.labels[i].shape[1],
                                  test_loader.dataset.num_classes], dtype=int))

    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs, cur_maps, cur_xs, cur_ys = data

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs, _ = net(inps_c)
            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)

            for j in range(soft_outs.shape[0]):
                cur_map = cur_maps[j]
                cur_x = cur_xs[j]
                cur_y = cur_ys[j]

                soft_outs_p = soft_outs.permute(0, 2, 3, 1).cpu().detach().numpy()

                prob_im[cur_map][cur_x:cur_x + test_loader.dataset.crop_size,
                                 cur_y:cur_y + test_loader.dataset.crop_size, :] += soft_outs_p[j, :, :, :]
                occur_im[cur_map][cur_x:cur_x + test_loader.dataset.crop_size,
                                  cur_y:cur_y + test_loader.dataset.crop_size, :] += 1

    for i, img in enumerate(test_loader.dataset.images):
        occur_im[i][np.where(occur_im[i] == 0)] = 1
        prob_im_argmax = np.argmax(prob_im[i] / occur_im[i].astype(float), axis=-1)
        print('test_full check', prob_im_argmax.shape, np.bincount(prob_im_argmax.flatten()),
              test_loader.dataset.labels[i].shape, np.bincount(test_loader.dataset.labels[i].flatten()))

        # pixel outside area should be 0 for the final image
        prob_im_argmax[np.where(test_loader.dataset.labels[i] == 2)] = 0
        # Saving predictions.
        imageio.imwrite(os.path.join(output_path, img + '_pred.png'), prob_im_argmax * 255)

        # filtering out pixels
        labs = test_loader.dataset.labels[i]
        coord = np.where(labs != 2)
        labs = labs[coord]
        prds = prob_im_argmax[coord]
        print(labs.shape, prds.shape, np.bincount(labs.flatten()), np.bincount(prds.flatten()))

        acc = accuracy_score(labs, prds)
        conf_m = confusion_matrix(labs, prds)
        f1_s_w = f1_score(labs, prds, average='weighted')
        f1_s_micro = f1_score(labs, prds, average='micro')
        f1_s_macro = f1_score(labs, prds, average='macro')
        kappa = cohen_kappa_score(labs, prds)
        jaccard = jaccard_score(labs, prds)
        tau, p = stats.kendalltau(labs, prds)
        prec_micro = precision_score(labs, prds, average='micro')
        prec_macro = precision_score(labs, prds, average='macro')
        prec_binary = precision_score(labs, prds, average='binary')
        rec_micro = recall_score(labs, prds, average='micro')
        rec_macro = recall_score(labs, prds, average='macro')
        rec_binary = recall_score(labs, prds, average='binary')

        _sum = 0.0
        for k in range(len(conf_m)):
            _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

        average_acc += acc
        average_n_acc += _sum / float(outs.shape[1])
        average_conf_m += conf_m
        average_f1_s_w += f1_s_w
        average_f1_s_micro += f1_s_micro
        average_f1_s_macro += f1_s_macro
        average_kappa += kappa
        average_jaccard += jaccard
        average_tau += tau
        average_prec_micro += prec_micro
        average_prec_macro += prec_macro
        average_prec_binary += prec_binary
        average_rec_micro += rec_micro
        average_rec_macro += rec_macro
        average_rec_binary += rec_binary

        print("---- Validation/Test -- Image: " + img + " -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
              " Precision micro= " + "{:.4f}".format(prec_micro) +
              " Precision macro= " + "{:.4f}".format(prec_macro) +
              " Precision binary= " + "{:.4f}".format(prec_binary) +
              " Recall micro= " + "{:.4f}".format(rec_micro) +
              " Recall macro= " + "{:.4f}".format(rec_macro) +
              " Recall binary= " + "{:.4f}".format(rec_binary) +
              " F1 score weighted= " + "{:.4f}".format(f1_s_w) +
              " F1 score micro= " + "{:.4f}".format(f1_s_micro) +
              " F1 score macro= " + "{:.4f}".format(f1_s_macro) +
              " Kappa= " + "{:.4f}".format(kappa) +
              " Jaccard= " + "{:.4f}".format(jaccard) +
              " Tau= " + "{:.4f}".format(tau) +
              " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
              )

    print("---- Validation/Test -- OVERALL -- Epoch " + str(epoch) +
          " -- Time " + str(datetime.datetime.now().time()) +
          " Overall Accuracy= " + "{:.4f}".format(average_acc / float(len(test_loader.dataset.images))) +
          " Normalized Accuracy= " + "{:.4f}".format(average_n_acc / float(len(test_loader.dataset.images))) +
          " Precision micro= " + "{:.4f}".format(average_prec_micro / float(len(test_loader.dataset.images))) +
          " Precision macro= " + "{:.4f}".format(average_prec_macro / float(len(test_loader.dataset.images))) +
          " Precision binary= " + "{:.4f}".format(average_prec_binary / float(len(test_loader.dataset.images))) +
          " Recall micro= " + "{:.4f}".format(average_rec_micro / float(len(test_loader.dataset.images))) +
          " Recall macro= " + "{:.4f}".format(average_rec_macro / float(len(test_loader.dataset.images))) +
          " Recall binary= " + "{:.4f}".format(average_rec_binary / float(len(test_loader.dataset.images))) +
          " F1 score weighted= " + "{:.4f}".format(average_f1_s_w / float(len(test_loader.dataset.images))) +
          " F1 score micro= " + "{:.4f}".format(average_f1_s_micro / float(len(test_loader.dataset.images))) +
          " F1 score macro= " + "{:.4f}".format(average_f1_s_macro / float(len(test_loader.dataset.images))) +
          " Kappa= " + "{:.4f}".format(average_kappa / float(len(test_loader.dataset.images))) +
          " Jaccard= " + "{:.4f}".format(average_jaccard / float(len(test_loader.dataset.images))) +
          " Tau= " + "{:.4f}".format(average_tau / float(len(test_loader.dataset.images))) +
          " Confusion Matrix= " + np.array_str(average_conf_m).replace("\n", "")
          )

    sys.stdout.flush()


def test(test_loader, net, epoch, loss_type):
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
            outs, _ = net(inps_c)

            if 'Binary' in loss_type:
                sigmoid_outs = torch.sigmoid(outs)
                # Obtaining predictions.
                prds = (sigmoid_outs > 0.5).int().cpu().data.numpy().flatten()
            else:
                soft_outs = F.softmax(outs, dim=1)
                # Obtaining predictions.
                prds = soft_outs.cpu().data.numpy().argmax(axis=1).flatten()
            labs = labs.flatten()

            # filtering out pixels
            coord = np.where(labs != 2)
            labs = labs[coord]
            prds = prds[coord]

            track_cm += confusion_matrix(labs, prds, labels=[0, 1])

        acc = (track_cm[0][0] + track_cm[1][1]) / np.sum(track_cm)
        f1_s = f1_with_cm(track_cm)
        kappa = kappa_with_cm(track_cm)
        jaccard = jaccard_with_cm(track_cm)

        _sum = 0.0
        for k in range(len(track_cm)):
            _sum += (track_cm[k][k] / float(np.sum(track_cm[k])) if np.sum(track_cm[k]) != 0 else 0)
        nacc = _sum / float(test_loader.dataset.num_classes)

        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(nacc) +
              " F1 Score= " + "{:.4f}".format(f1_s) +
              " Kappa= " + "{:.4f}".format(kappa) +
              " Jaccard= " + "{:.4f}".format(jaccard) +
              " Confusion Matrix= " + np.array_str(track_cm).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, nacc, f1_s, kappa, track_cm


def train(train_loader, net, criterion, optimizer, epoch, loss_type):
    # Setting network for training mode.
    net.train()

    # Average Meter for batch loss.
    train_loss = list()

    # Iterating over batches.
    for i, data in enumerate(train_loader):
        # Obtaining buzz sounds and labels
        inps, labels = data[0], data[1]

        # if there is only one class
        if len(np.bincount(labels.flatten())) == 1:
            continue

        # Casting to cuda variables.
        inps = Variable(inps).cuda()
        labs = Variable(labels).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs, _ = net(inps)

        # Computing Cross entropy loss.
        if 'Binary' in loss_type:
            loss = criterion(outs.squeeze(), labs.float())
        else:
            loss = criterion(outs, labs)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            if 'Binary' in loss_type:
                sigmoid_outs = torch.sigmoid(outs)
                # Obtaining predictions.
                prds = (sigmoid_outs > 0.5).int().cpu().data.numpy().flatten()
            else:
                soft_outs = F.softmax(outs, dim=1)
                # Obtaining predictions.
                prds = soft_outs.cpu().data.numpy().argmax(axis=1).flatten()
            labels = labels.cpu().data.numpy().flatten()

            # filtering out pixels
            coord = np.where(labels != 2)
            labels = labels[coord]
            prds = prds[coord]

            acc = accuracy_score(labels, prds)
            conf_m = confusion_matrix(labels, prds)
            f1_s = f1_score(labels, prds, average='weighted')

            _sum = 0.0
            for k in range(len(conf_m)):
                _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i + 1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / float(train_loader.dataset.num_classes)) +
                  " F1 Score= " + "{:.4f}".format(f1_s) +
                  " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )
            sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True, help='Operation to be performed]',
                        choices=['Train', 'Test', 'Plot', 'Full_Test'])
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save outcomes (such as images and trained models) of the algorithm.')

    # dataset options
    parser.add_argument('--dataset', type=str, required=True, help='Dataset.',
                        choices=['River', 'Orange', 'Coffee', 'Coffee_Full', 'Road', 'Coffee_Crop', 'Tree'])
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--training_images', type=str, nargs="+", required=False, help='Training image names.')
    parser.add_argument('--testing_images', type=str, nargs="+", required=False, help='Testing image names.')
    parser.add_argument('--crop_size', type=int, required=True, help='Crop size.')
    parser.add_argument('--stride_crop', type=int, required=True, help='Stride size')

    # model options
    parser.add_argument('--model', type=str, required=True, default=None,
                        help='Model to be used.', choices=['WideResNet', 'EfficientNetB0', 'DenseNet121'])
    parser.add_argument('--loss', type=str, required=True, default=None, help='Loss function to be used.',
                        choices=['CE', 'BinaryCE', 'BinaryFocal', 'Focal', 'UnifiedFocal', 'DualFocal'])
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model that can be load and used for inference.')
    parser.add_argument('--weights', type=float, nargs='+', default=[1.0, 1.0], help='Weight Loss.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=50, help='Number of epochs')

    parser.add_argument('--weight_sampler', type=str2bool, default=False, help='Use weight sampler for loader?')
    parser.add_argument('--crop', type=str2bool, default=False, help='River crop dataset?')
    args = parser.parse_args()
    print(sys.argv[0], args)

    # data loaders
    if args.operation == 'Train':
        if args.dataset == 'River':
            print('---- training data ----')
            train_dataset = DataLoader('Full_train', args.dataset, args.dataset_path, args.training_images,
                                       args.crop_size, args.stride_crop, output_path=args.output_path, crop=args.crop)
            print('---- testing data ----')
            test_dataset = DataLoader('Full_test', args.dataset, args.dataset_path, args.testing_images,
                                      args.crop_size, args.stride_crop,  # args.crop_size,
                                      mean=train_dataset.mean, std=train_dataset.std, crop=args.crop)
        elif args.dataset == 'Road':
            print('---- training data ----')
            train_dataset = DataLoaderRoad('Train', args.dataset, args.dataset_path, args.training_images,
                                           args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderRoad('Test', args.dataset, args.dataset_path, args.testing_images,
                                          args.crop_size, args.crop_size,
                                          mean=train_dataset.mean, std=train_dataset.std)
        elif args.dataset == 'Tree':
            print('---- training data ----')
            train_dataset = DataLoaderTree('Train', args.dataset, args.dataset_path, args.training_images,
                                           args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderTree('Test', args.dataset, args.dataset_path, args.testing_images,
                                          args.crop_size, args.stride_crop,  # args.crop_size,
                                          mean=train_dataset.mean, std=train_dataset.std)
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
            train_dataset = DataLoaderCoffeeFull('Full_Train', args.dataset, args.dataset_path, args.training_images,
                                                 args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderCoffeeFull('Full_Test', args.dataset, args.dataset_path, args.testing_images,
                                                args.crop_size, args.stride_crop,
                                                mean=train_dataset.mean, std=train_dataset.std)
        elif args.dataset == 'Coffee_Crop':
            print('---- training data ----')
            train_dataset = DataLoaderCoffeeCrop('Train', args.dataset, args.dataset_path, args.training_images,
                                                 args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderCoffeeCrop('Test', args.dataset, args.dataset_path, args.testing_images,
                                                args.crop_size, args.crop_size,
                                                mean=train_dataset.mean, std=train_dataset.std)
        else:
            raise NotImplementedError("Dataset " + args.dataset + " not implemented")

        if args.weight_sampler is False:
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                           shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
        else:
            train_dataloader = sample_weight_train_loader(train_dataset, train_dataset.gen_classes, args.batch_size)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # network
        if args.model == 'WideResNet':
            model = FCNWideResNet50(1 if 'Binary' in args.loss else train_dataset.num_classes,
                                    pretrained=True, skip_layers='2_4', classif=True)
        elif args.model == 'DenseNet121':
            model = FCNDenseNet121(train_dataset.num_classes, pretrained=True, skip_layers='1_2_3_4', classif=True)
        elif args.model == 'EfficientNetB0':
            model = FCNEfficientNetB0(1 if 'Binary' in args.loss else train_dataset.num_classes,
                                      pretrained=True, classif=True)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")
        # model.cuda()

        # loss
        if args.loss == 'CE':
            # ignore_index=2 because of the background of the Coffee_Full dataset
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.weights), ignore_index=2).cuda()
        elif args.loss == 'BinaryCE':
            # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837/2
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(args.weights[1:])).cuda()
        elif args.loss == 'BinaryFocal':
            criterion = BinaryFocalLoss(alpha=args.weights[1], gamma=2).cuda()
        elif args.loss == 'Focal':
            criterion = FocalLossV2(weight=torch.FloatTensor(args.weights).cuda(), gamma=2).cuda()
        elif args.loss == 'DualFocal':
            criterion = DualFocalLoss(alpha=1, beta=1, gamma=1, rho=1).cuda()
        elif args.loss == 'UnifiedFocal':
            # original
            # criterion = UnifiedFocalLoss(weight=torch.FloatTensor(args.weights).cuda(), delta=0.5, gamma=2).cuda()
            # paper
            criterion = UnifiedFocalLoss(internal_weight=0.5, weight=torch.FloatTensor(args.weights).cuda(),
                                         delta=0.6, gamma=2).cuda()
        else:
            raise NotImplementedError("Loss " + args.loss + " not implemented")
        # tl_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        # if model.classif is True:
        #     if args.model == 'WideResNet':
        #         optimizer = optim.Adam([
        #             {'params': list(model.parameters())[:-6]},
        #             {'params': list(model.parameters())[-6:], 'lr': args.learning_rate, 'weight_decay': args.weight_decay}],
        #             lr=args.learning_rate/10, weight_decay=args.weight_decay, betas=(0.9, 0.99)
        #         )
        #     elif args.model == 'EfficientNetB0':
        #         optimizer = optim.Adam([
        #             {'params': list(model.parameters())[:-10]},
        #             {'params': list(model.parameters())[-10:], 'lr': args.learning_rate, 'weight_decay': args.weight_decay}],
        #             lr=args.learning_rate / 10, weight_decay=args.weight_decay, betas=(0.9, 0.99)
        #         )
        #     else:
        #         raise NotImplementedError("Network " + args.model + " not implemented")
        # else:
        #     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        #                            betas=(0.9, 0.99))
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                               betas=(0.9, 0.99))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)  # original 50, epoch 500
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
            train(train_dataloader, model, criterion, optimizer, epoch, args.loss)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                acc, nacc, f1_s, kappa, _ = test(test_dataloader, model, epoch, args.loss)
                save_best_models(model, optimizer, args.output_path, best_records, epoch, kappa)

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
            model = FCNWideResNet50(test_dataset.num_classes, pretrained=True, classif=True)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        epoch = 0
        if args.model_path is not None:
            model.load_state_dict(torch.load(args.model_path))
            epoch = int(os.path.basename(args.model_path)[:-4].split('_')[-1])
        model.cuda()

        test(test_dataloader, model, epoch, args.loss)
    elif args.operation == 'Full_Test':
        print('---- testing ----')
        # assert args.model_path is not None, "For inference, flag --model_path should be set."

        if args.model_path is None:
            print('loading from best_records')
            best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
            index = 0
            for i in range(len(best_records)):
                if best_records[index]['kappa'] < best_records[i]['kappa']:
                    index = i
            epoch = int(best_records[index]['epoch'])
            cur_model = 'model_' + str(epoch) + '.pth'
        else:
            print('loading from args.model_path')
            epoch = int(args.model_path[:-4].split('_')[-1])
            cur_model = args.model_path

        if args.dataset == 'River':
            test_dataset = DataLoader('Full_test', args.dataset, args.dataset_path, args.testing_images,
                                      args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Road':
            test_dataset = DataLoaderRoad('Test', args.dataset, args.dataset_path, args.testing_images,
                                          args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Orange':
            test_dataset = DataLoaderOrange('Test', args.dataset, args.dataset_path, args.crop_size, args.stride_crop,
                                            output_path=args.output_path)
        elif args.dataset == 'Coffee_Full':
            test_dataset = DataLoaderCoffeeFull('Full_Test', args.dataset, args.dataset_path, args.testing_images,
                                                args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Coffee_Crop':
            test_dataset = DataLoaderCoffeeCrop('Test', args.dataset, args.dataset_path, args.testing_images,
                                                args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Tree':
            test_dataset = DataLoaderTree('Test', args.dataset, args.dataset_path, args.testing_images,
                                          args.crop_size, args.stride_crop, output_path=args.output_path)
        else:
            raise NotImplementedError("Dataset " + args.dataset + " not implemented")

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # network
        if args.model == 'WideResNet':
            model = FCNWideResNet50(test_dataset.num_classes, pretrained=True, classif=True)
        elif args.model == 'DenseNet121':
            model = FCNDenseNet121(test_dataset.num_classes, pretrained=True, skip_layers='1_2_3_4', classif=True)
        elif args.model == 'EfficientNetB0':
            model = FCNEfficientNetB0(1 if 'Binary' in args.loss else test_dataset.num_classes,
                                      pretrained=True, classif=True)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        print("Loading model " + cur_model)
        model.load_state_dict(torch.load(os.path.join(args.output_path, cur_model)))
        model.cuda()

        test_full_map(test_dataloader, model, epoch, args.output_path)
    else:
        raise NotImplementedError("Operation " + args.operation + " not implemented")
