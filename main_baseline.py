import gc
import sys
import datetime
import numpy as np
import imageio
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dataloader import DataLoader
from dataloader_orange import DataLoaderOrange
from dataloader_coffee import DataLoaderCoffee

from config import *
from utils import *
from network import FCNWideResNet50
from focal_loss import BinaryFocalLoss, FocalLoss, FocalLossV2


def test_full_map(test_loader, net, epoch, output_path):
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
            outs, _, _ = net(inps_c)
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
    print(prob_im_argmax.shape, np.bincount(prob_im_argmax.flatten()),
          test_loader.dataset.labels[0].shape, np.bincount(test_loader.dataset.labels[0].flatten()))

    # Saving predictions.
    imageio.imwrite(os.path.join(output_path, 'baseline_prd.png'), prob_im_argmax*255)

    acc = accuracy_score(test_loader.dataset.labels[0].flatten(), prob_im_argmax.flatten())
    conf_m = confusion_matrix(test_loader.dataset.labels[0].flatten(), prob_im_argmax.flatten())

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

    return 0, 0, 0


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
            outs, _, _ = net(inps_c)

            if 'Binary' in loss_type:
                sigmoid_outs = torch.sigmoid(outs)
                # Obtaining predictions.
                prds = (sigmoid_outs > 0.5).int().cpu().data.numpy()
            else:
                soft_outs = F.softmax(outs, dim=1)
                # Obtaining predictions.
                prds = soft_outs.cpu().data.numpy().argmax(axis=1)

            track_cm += confusion_matrix(labs.flatten(), prds.flatten(), labels=[0, 1])

        acc = (track_cm[0][0] + track_cm[1][1]) / np.sum(track_cm)
        f1_s = f1_with_cm(track_cm)

        _sum = 0.0
        for k in range(len(track_cm)):
            _sum += (track_cm[k][k] / float(np.sum(track_cm[k])) if np.sum(track_cm[k]) != 0 else 0)
        nacc = _sum / float(test_loader.dataset.num_classes)

        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(nacc) +
              " F1 Score= " + "{:.4f}".format(f1_s) +
              " Confusion Matrix= " + np.array_str(track_cm).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, nacc, track_cm


def train(train_loader, net, criterion, optimizer, epoch, loss_type):
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
        outs, _, _ = net(inps)

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
                prds = (sigmoid_outs > 0.5).int().cpu().data.numpy()
            else:
                soft_outs = F.softmax(outs, dim=1)
                # Obtaining predictions.
                prds = soft_outs.cpu().data.numpy().argmax(axis=1)
            labels = labels.cpu().data.numpy()
            acc = accuracy_score(labels.flatten(), prds.flatten())
            conf_m = confusion_matrix(labels.flatten(), prds.flatten())
            f1_s = f1_score(labels.flatten(), prds.flatten(), average='weighted')

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
                        choices=['River', 'Orange', 'Coffee'])
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--training_images', type=str, nargs="+", required=False, help='Training image names.')
    parser.add_argument('--testing_images', type=str, nargs="+", required=False, help='Testing image names.')
    parser.add_argument('--crop_size', type=int, required=True, help='Crop size.')
    parser.add_argument('--stride_crop', type=int, required=True, help='Stride size')

    # model options
    parser.add_argument('--model', type=str, required=True, default=None,
                        help='Model to be used.', choices=['WideResNet'])
    parser.add_argument('--loss', type=str, required=True, default=None,
                        help='Loss function to be used.', choices=['CE', 'BinaryCE', 'BinaryFocal', 'Focal'])
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model that can be load and used for inference.')
    parser.add_argument('--weights', type=float, nargs='+', default=[1.0, 1.0], help='Weight Loss.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()
    print(args)

    # data loaders
    if args.operation == 'Train':
        if args.dataset == 'River':
            print('---- training data ----')
            train_dataset = DataLoader('Train', args.dataset_path, args.training_images, args.crop_size,
                                       args.stride_crop, output_path=args.output_path)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                           shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
            print('---- testing data ----')
            test_dataset = DataLoader('Full_test', args.dataset_path, args.testing_images,
                                      args.crop_size, args.stride_crop, mean=train_dataset.mean, std=train_dataset.std)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                          shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
        elif args.dataset == 'Orange':
            print('---- training data ----')
            train_dataset = DataLoaderOrange('Train', args.dataset_path, args.crop_size, args.stride_crop,
                                             output_path=args.output_path)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                           shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
            print('---- testing data ----')
            test_dataset = DataLoaderOrange('Test', args.dataset_path, args.crop_size, args.stride_crop,
                                            mean=train_dataset.mean, std=train_dataset.std)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                          shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
        elif args.dataset == 'Coffee':
            print('---- training data ----')
            train_dataset = DataLoaderCoffee('Train', args.dataset, args.dataset_path, args.training_images,
                                             args.crop_size, args.stride_crop, output_path=args.output_path)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                           shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
            print('---- testing data ----')
            test_dataset = DataLoaderCoffee('Test', args.dataset, args.dataset_path, args.testing_images,
                                            args.crop_size, args.stride_crop,
                                            mean=train_dataset.mean, std=train_dataset.std)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                          shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
        else:
            raise NotImplementedError("Dataset " + args.dataset + " not implemented")

        # network
        if args.model == 'WideResNet':
            model = FCNWideResNet50(1 if 'Binary' in args.loss else train_dataset.num_classes,
                                    pretrained=True, classif=True)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")
        model.cuda()

        # loss
        if args.loss == 'CE':
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.weights)).cuda()
        elif args.loss == 'BinaryCE':
            # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837/2
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(args.weights[1:])).cuda()
        elif args.loss == 'BinaryFocal':
            criterion = BinaryFocalLoss(alpha=args.weights[1], gamma=2).cuda()
        elif args.loss == 'Focal':
            criterion = FocalLossV2(weight=torch.FloatTensor(args.weights).cuda(), gamma=2).cuda()
        else:
            raise NotImplementedError("Loss " + args.loss + " not implemented")
        # tl_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        if model.classif is True:
            optimizer = optim.Adam([
                {'params': list(model.parameters())[:-10]},
                {'params': list(model.parameters())[-10:], 'lr': args.learning_rate, 'weight_decay': args.weight_decay}],
                lr=args.learning_rate/10, weight_decay=args.weight_decay, betas=(0.9, 0.99)
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                   betas=(0.9, 0.99))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

        curr_epoch = 1
        best_records = []
        print('---- training ----')
        for epoch in range(curr_epoch, args.epoch_num + 1):
            train(train_dataloader, model, criterion, optimizer, epoch, args.loss)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                acc, nacc, _ = test(test_dataloader, model, epoch, args.loss)
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
            model = FCNWideResNet50(test_dataset.num_classes, pretrained=True, classif=True)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        epoch = 0
        if args.model_path is not None:
            model.load_state_dict(torch.load(args.model_path))
            epoch = int(os.path.basename(args.model_path)[:-4].split('_')[-1])
        model.cuda()

        test(test_dataloader, model, epoch)
    elif args.operation == 'Full_Test':
        print('---- testing ----')
        # assert args.model_path is not None, "For inference, flag --model_path should be set."

        test_dataset = DataLoader('Full_test', args.dataset_path, args.testing_images,
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

        test_full_map(test_dataloader, model, epoch, args.output_path)
    else:
        raise NotImplementedError("Operation " + args.operation + " not implemented")
