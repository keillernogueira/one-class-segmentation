import gc
import os
import sys
import datetime
import imageio
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, \
    jaccard_score, precision_score, recall_score
import scipy.stats as stats

import torch
from torch import optim
from torch.autograd import Variable

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dataloaders.dataloader import DataLoader
from dataloaders.dataloader_road import DataLoaderRoad
from dataloaders.dataloader_orange import DataLoaderOrange
from dataloaders.dataloader_coffee import DataLoaderCoffee
from dataloaders.dataloader_coffee_full import DataLoaderCoffeeFull
from dataloaders.dataloader_coffee_crop import DataLoaderCoffeeCrop
from dataloaders.dataloader_tree import DataLoaderTree
from dataloaders.dataloader_5billion import DataLoader5Billion
from dataloaders.isprs_dataloader import ISPRSDataLoader

from config import *
from utils import *
from networks.FCNWideResNet50 import FCNWideResNet50
from networks.efficientnet import FCNEfficientNetB0
from networks.FCNDenseNet121 import FCNDenseNet121
from networks.unet import UNet

from feat_ext import general_feature_extractor
from contrastive_loss import ContrastiveLoss
from contrastive_loss_double_margin import ContrastiveLossDoubleMargin
from learnt_prototypical import LearntPrototypes


def test_full_map(test_loader, criterion, net, epoch, output_path):
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
        prob_im.append(np.zeros([test_loader.dataset.labels[i].shape[0],
                                 test_loader.dataset.labels[i].shape[1]], dtype=np.float32))
        occur_im.append(np.zeros([test_loader.dataset.labels[i].shape[0],
                                  test_loader.dataset.labels[i].shape[1]], dtype=int))

    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs, cur_maps, cur_xs, cur_ys = data[0], data[1], data[2], data[3], data[4]

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs = net(inps_c)
            b, c, h, w = outs.shape

            # max because the values are negative
            outs_max, _ = outs.max(1)
            outs_max_neg = -outs_max.cpu().detach().numpy()

            for j in range(outs.shape[0]):
                cur_map = cur_maps[j]
                cur_x = cur_xs[j]
                cur_y = cur_ys[j]

                prob_im[cur_map][cur_x:cur_x + test_loader.dataset.crop_size,
                                 cur_y:cur_y + test_loader.dataset.crop_size] += outs_max_neg[j, :, :]
                occur_im[cur_map][cur_x:cur_x + test_loader.dataset.crop_size,
                                  cur_y:cur_y + test_loader.dataset.crop_size] += 1

    for i, img in enumerate(test_loader.dataset.images):
        # normalise to remove non-predicted pixels
        occur_im[i][np.where(occur_im[i] == 0)] = 1

        if hasattr(criterion, 'pos_margin'):
            prob_im_argmax = ((prob_im[i] / occur_im[i].astype(float)) < criterion.pos_margin).astype(int)
        else:
            prob_im_argmax = ((prob_im[i] / occur_im[i].astype(float)) < criterion.margin).astype(int)
        print(prob_im_argmax.shape, np.bincount(prob_im_argmax.flatten()),
              test_loader.dataset.labels[i].shape, np.bincount(test_loader.dataset.labels[i].flatten()))

        # pixel outside area should be 0 for the final image
        prob_im_argmax[np.where(test_loader.dataset.labels[i] == 2)] = 0
        # Saving predictions.
        imageio.imwrite(os.path.join(output_path, img + '_pred.png'), prob_im_argmax*255)

        if test_loader.dataset.dataset == 'Coffee_Full' or test_loader.dataset.dataset == 'Orange':
            labs = test_loader.dataset.labels[i]
            if test_loader.dataset.dataset == 'Orange':  # removing training part
                labs = labs[5051:, :]
                prob_im_argmax = prob_im_argmax[5051:, :]
            coord = np.where(labs != 2)
            lbl = labs[coord]
            pred = prob_im_argmax[coord]
        else:
            lbl = test_loader.dataset.labels[i].flatten()
            pred = prob_im_argmax.flatten()

        print(lbl.shape, np.bincount(lbl.flatten()), pred.shape, np.bincount(pred.flatten()))

        acc = accuracy_score(lbl, pred)
        conf_m = confusion_matrix(lbl, pred)
        f1_s_w = f1_score(lbl, pred, average='weighted')
        f1_s_micro = f1_score(lbl, pred, average='micro')
        f1_s_macro = f1_score(lbl, pred, average='macro')
        kappa = cohen_kappa_score(lbl, pred)
        jaccard = jaccard_score(lbl, pred)
        tau, p = stats.kendalltau(lbl, pred)
        prec_micro = precision_score(lbl, pred, average='micro')
        prec_macro = precision_score(lbl, pred, average='macro')
        prec_binary = precision_score(lbl, pred, average='binary')
        rec_micro = recall_score(lbl, pred, average='micro')
        rec_macro = recall_score(lbl, pred, average='macro')
        rec_binary = recall_score(lbl, pred, average='binary')

        _sum = 0.0
        for k in range(len(conf_m)):
            _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

        average_acc += acc
        average_n_acc += _sum / float(2.0)
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
              " Normalized Accuracy= " + "{:.4f}".format(_sum / float(2.0)) +
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


def test(test_loader, criterion, net, epoch):
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

            # pred = (-outs <= 0.999).int().detach().cpu().numpy()  # v1
            b, c, h, w = outs.shape
            # max because the values are negative
            outs, _ = outs.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c).max(1)
            if hasattr(criterion, 'pos_margin'):
                pred = (-outs < criterion.pos_margin).int().detach().cpu().numpy().flatten()
            else:
                pred = (-outs < criterion.margin).int().detach().cpu().numpy().flatten()
            labs = labs.flatten()

            # filtering out pixels
            coord = np.where(labs != 2)
            labs = labs[coord]
            pred = pred[coord]

            track_cm += confusion_matrix(labs, pred, labels=[0, 1])

    acc = (track_cm[0][0] + track_cm[1][1])/np.sum(track_cm)
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


def train(train_loader, net, criterion, optimizer, epoch, output):
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
            b, c, h, w = outs.shape
            # max because the values are negative
            outs, _ = outs.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c).max(1)
            if hasattr(criterion, 'pos_margin'):
                pred = (-outs < criterion.pos_margin).int().detach().cpu().numpy().flatten()
            else:
                pred = (-outs < criterion.margin).int().detach().cpu().numpy().flatten()
            labels = labels.detach().cpu().numpy().flatten()

            # filtering out pixels
            coord = np.where(labels != 2)
            labels = labels[coord]
            pred = pred[coord]

            # checking distances
            # n_outs = -outs.detach().cpu().numpy().flatten()[coord]
            # false_pos = n_outs[np.logical_and(labels == 0, pred == 1)]
            # false_neg = n_outs[np.logical_and(labels == 1, pred == 0)]
            # print(false_pos.shape, false_pos.shape[0])
            # if false_pos.shape[0] != 0:
            #     print('false_pos', np.min(false_pos), np.mean(false_pos), np.max(false_pos))
            # if false_neg.shape[0] != 0:
            #     print('false_neg', np.min(false_neg), np.mean(false_neg), np.max(false_neg))

            # metrics
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
                        choices=['River', 'Orange', 'Coffee', 'Coffee_Full', '5Billion',
                                 'Coffee_Crop', 'Road', 'Tree', 'Vaihingen'])
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--training_images', type=str, nargs="+", required=False, help='Training image names.')
    parser.add_argument('--testing_images', type=str, nargs="+", required=False, help='Testing image names.')
    parser.add_argument('--crop_size', type=int, required=True, help='Crop size.')
    parser.add_argument('--stride_crop', type=int, required=True, help='Stride size')

    # model options
    parser.add_argument('--model', type=str, required=True, default=None,
                        help='Model to be used.', choices=['WideResNet', 'WideResNet_4', 'UNet',
                                                           'DenseNet121', 'EfficientNetB0'])
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
    parser.add_argument('--weight_sampler', type=str2bool, default=False, help='Use weight sampler for loader?')
    parser.add_argument('--num_prototypes', type=int, default=1, help='Number of prototypes')
    args = parser.parse_args()
    print(sys.argv[0], args)

    # data loaders
    if args.operation == 'Train':
        if args.dataset == 'River':
            print('---- training data ----')
            train_dataset = DataLoader('Full_train', args.dataset, args.dataset_path, args.training_images,
                                       args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoader('Full_test', args.dataset, args.dataset_path, args.testing_images,
                                      args.crop_size, args.crop_size,  # args.stride_crop,
                                      mean=train_dataset.mean, std=train_dataset.std)
        elif args.dataset == 'Vaihingen':
            print('---- training data ----')
            train_dataset = ISPRSDataLoader('Train', args.dataset, args.dataset_path, args.training_images,
                                            args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = ISPRSDataLoader('Validation', args.dataset, args.dataset_path, args.testing_images,
                                           args.crop_size, args.crop_size,
                                           mean=train_dataset.mean, std=train_dataset.std)
        elif args.dataset == '5Billion':
            print('---- training data ----')
            train_dataset = DataLoader5Billion('Full_train', args.dataset, args.dataset_path, args.training_images,
                                               args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoader5Billion('Full_test', args.dataset, args.dataset_path, args.testing_images,
                                              args.crop_size, args.crop_size,
                                              mean=train_dataset.mean, std=train_dataset.std)
        elif args.dataset == 'Road':
            print('---- training data ----')
            train_dataset = DataLoaderRoad('Train', args.dataset, args.dataset_path, args.training_images,
                                           args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderRoad('Test', args.dataset, args.dataset_path, args.testing_images,
                                          args.crop_size, args.crop_size,
                                          mean=train_dataset.mean, std=train_dataset.std)
        elif args.dataset == 'Coffee_Crop':
            print('---- training data ----')
            train_dataset = DataLoaderCoffeeCrop('Train', args.dataset, args.dataset_path, args.training_images,
                                                 args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderCoffeeCrop('Test', args.dataset, args.dataset_path, args.testing_images,
                                                args.crop_size, args.crop_size,
                                                mean=train_dataset.mean, std=train_dataset.std)
        elif args.dataset == 'Orange':
            print('---- training data ----')
            train_dataset = DataLoaderOrange('Train', args.dataset, args.dataset_path, args.crop_size, args.stride_crop,
                                             output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderOrange('Test', args.dataset, args.dataset_path,
                                            args.crop_size, args.crop_size,  # args.stride_crop,
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
        elif args.dataset == 'Tree':
            print('---- training data ----')
            train_dataset = DataLoaderTree('Train', args.dataset, args.dataset_path, args.training_images,
                                           args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderTree('Test', args.dataset, args.dataset_path, args.testing_images,
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

        # # network
        # if args.model == 'WideResNet':
        #     model = LearntPrototypes(FCNWideResNet50(train_dataset.num_classes, pretrained=True, classif=False),
        #                              squared=False, n_prototypes=2, embedding_dim=2560)
        # elif args.model == 'EfficientNetB0':
        #     model = LearntPrototypes(FCNEfficientNetB0(train_dataset.num_classes, pretrained=True, classif=False),
        #                              squared=False, n_prototypes=2, embedding_dim=2096)
        # else:
        #     raise NotImplementedError("Network " + args.model + " not implemented")
        # network
        if args.model == 'WideResNet':
            model = LearntPrototypes(FCNWideResNet50(train_dataset.num_classes, pretrained=True,
                                                     skip_layers='2_4', classif=False),
                                     squared=False, n_prototypes=args.num_prototypes, embedding_dim=2560)  # original
        elif args.model == 'WideResNet_4':
            model = LearntPrototypes(FCNWideResNet50(train_dataset.num_classes, pretrained=True,
                                                     skip_layers='1_2_3_4', classif=False),
                                     squared=True, n_prototypes=args.num_prototypes, embedding_dim=3840)
        elif args.model == 'DenseNet121':
            model = LearntPrototypes(FCNDenseNet121(train_dataset.num_classes, pretrained=True,
                                                    skip_layers='1_2_3_4', classif=False),
                                     squared=True, n_prototypes=args.num_prototypes, embedding_dim=1920)
        elif args.model == 'UNet':
            model = LearntPrototypes(UNet(train_dataset.num_classes, input_channels=3,
                                          skip_layers='1_2_3_4', classif=False),
                                     squared=True, n_prototypes=args.num_prototypes, embedding_dim=512)
        elif args.model == 'EfficientNetB0':
            model = LearntPrototypes(FCNEfficientNetB0(train_dataset.num_classes, pretrained=True, classif=False),
                                     squared=True, n_prototypes=args.num_prototypes, embedding_dim=2096)
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
                # acc, nacc, f1_s, kappa, track_cm
                acc, nacc, f1_s, kappa, _ = test(test_dataloader, criterion, model, epoch)
                save_best_models(model, optimizer, args.output_path, best_records, epoch, kappa)
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

        print('---- data ----')
        if args.dataset == 'River':
            test_dataset = DataLoader('Full_test', args.dataset, args.dataset_path, args.testing_images,
                                      args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Vaihingen':
            test_dataset = ISPRSDataLoader('Validation', args.dataset, args.dataset_path, args.testing_images,
                                           args.crop_size, args.crop_size, output_path=args.output_path)
        elif args.dataset == '5Billion':
            test_dataset = DataLoader5Billion('Full_test', args.dataset, args.dataset_path, args.testing_images,
                                              args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Tree':
            test_dataset = DataLoaderTree('Test', args.dataset, args.dataset_path, args.testing_images,
                                          args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Orange':
            test_dataset = DataLoaderOrange('Test', args.dataset, args.dataset_path, args.crop_size, args.stride_crop,
                                            output_path=args.output_path)
        elif args.dataset == 'Coffee':
            test_dataset = DataLoaderCoffee('Test', args.dataset, args.dataset_path, args.testing_images,
                                            args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Coffee_Full':
            test_dataset = DataLoaderCoffeeFull('Full_Test', args.dataset, args.dataset_path, args.testing_images,
                                                args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Coffee_Crop':
            test_dataset = DataLoaderCoffeeCrop('Test', args.dataset, args.dataset_path, args.testing_images,
                                                args.crop_size, args.stride_crop, output_path=args.output_path)
        elif args.dataset == 'Road':
            test_dataset = DataLoaderRoad('Test', args.dataset, args.dataset_path, args.testing_images,
                                          args.crop_size, args.stride_crop, output_path=args.output_path)
        else:
            raise NotImplementedError("Dataset " + args.dataset + " not implemented")

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # network
        if args.model == 'WideResNet':
            model = LearntPrototypes(FCNWideResNet50(test_dataset.num_classes, pretrained=True,
                                                     skip_layers='2_4', classif=False),
                                     squared=False, n_prototypes=args.num_prototypes, embedding_dim=2560)  # original
        elif args.model == 'WideResNet_4':
            model = LearntPrototypes(FCNWideResNet50(test_dataset.num_classes, pretrained=True,
                                                     skip_layers='1_2_3_4', classif=False),
                                     squared=True, n_prototypes=args.num_prototypes, embedding_dim=3840)
        elif args.model == 'DenseNet121':
            model = LearntPrototypes(FCNDenseNet121(test_dataset.num_classes, pretrained=True,
                                                    skip_layers='1_2_3_4', classif=False),
                                     squared=True, n_prototypes=args.num_prototypes, embedding_dim=1920)
        elif args.model == 'UNet':
            model = LearntPrototypes(UNet(test_dataset.num_classes, input_channels=3,
                                          skip_layers='1_2_3_4', classif=False),
                                     squared=True, n_prototypes=args.num_prototypes, embedding_dim=512)
        elif args.model == 'EfficientNetB0':
            model = LearntPrototypes(FCNEfficientNetB0(test_dataset.num_classes, pretrained=True, classif=False),
                                     squared=True, n_prototypes=args.num_prototypes, embedding_dim=2096)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        # loss
        criterion = ContrastiveLoss(args.margin, args.miner, args.weights)

        print("loading model_" + str(epoch) + '.pth')
        model.load_state_dict(torch.load(os.path.join(args.output_path, cur_model)))
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
