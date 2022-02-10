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

from dataloaders.dataloader import DataLoader
from config import *
from utils import *
from network import FCNWideResNet50


def test(test_loader, net, epoch):
    # Setting network for evaluation mode.
    net.eval()

    all_labels = None
    all_preds = None
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
            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)

            # Obtaining prior predictions.
            prds = soft_outs.cpu().data.numpy().argmax(axis=1)

            if all_labels is None:
                all_labels = labs
                all_preds = prds
            else:
                all_labels = np.concatenate((all_labels, labs))
                all_preds = np.concatenate((all_preds, prds))

        print('all', all_labels.shape, all_preds.shape)
        acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
        conf_m = confusion_matrix(all_labels.flatten(), all_preds.flatten(), labels=[0, 1])

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

    return acc, _sum / float(outs.shape[1]), conf_m


def train(train_loader, net, ce_criterion, tl_criterion, optimizer, epoch):
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
        outs, fv2, fv4 = net(inps)  # output here are the logits and 2 feature vectors

        # Concatenating features
        feat_flat = torch.cat([fv2, fv4], 1)
        feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(-1, feat_flat.size(1))

        # Online triplet mining and triplet loss
        a, p, n, aux = get_triples(feat_flat, labs.view(-1))
        loss_tl = tl_criterion(a, p, n)

        # Computing Cross entropy loss.
        loss_ce = ce_criterion(outs, labs)

        # Combining loss
        loss = loss_tl + loss_ce

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            # Obtaining predictions.
            soft_outs = F.softmax(outs, dim=1)
            prds = soft_outs.cpu().data.numpy().argmax(axis=1)

            acc = accuracy_score(labels.flatten(), prds.flatten())
            conf_m = confusion_matrix(labels.flatten(), prds.flatten(), labels=[0, 1])

            _sum = 0.0
            for k in range(len(conf_m)):
                _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i+1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
                  " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )
            sys.stdout.flush()

    gc.collect()


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

    # specific parameters
    parser.add_argument('--margin', type=float, default=1.0, help='Triplet loss margin')
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
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16,  # args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # network
        if args.model == 'WideResNet':
            model = FCNWideResNet50(train_dataset.num_classes, pretrained=True, skip=True, classif=True)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        # loss
        ce_criterion = nn.CrossEntropyLoss().cuda()
        tl_criterion = nn.TripletMarginLoss(margin=args.margin, p=2).cuda()

        # optimizing all parameters!
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
            train(train_dataloader, model, ce_criterion, tl_criterion, optimizer, epoch)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                acc, nacc, conf = test(test_dataloader, model, epoch)
                save_best_models(model, optimizer, args.output_path, best_records, epoch, nacc)

            scheduler.step()
    elif args.operation == 'Test':
        print('---- testing ----')
        # assert args.model_path is not None, "For inference, flag --model_path should be set."

        print('---- knn data ----')
        knn_dataset = DataLoader('KNN', args.dataset_path, args.training_images, args.crop_size,
                                 args.stride_crop, output_path=args.output_path)
        knn_dataloader = torch.utils.data.DataLoader(knn_dataset, batch_size=16,
                                                     shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
        print('---- testing data ----')
        test_dataset = DataLoader('Validation', args.dataset_path, args.testing_images,
                                  args.crop_size, args.stride_crop, output_path=args.output_path)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16,
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

        test(test_dataloader, model, epoch)
    else:
        raise NotImplementedError("Operation " + args.operation + " not implemented")
