import gc
import sys
import datetime

from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from dataloader import DataLoader
from config import *
from utils import *
from network import FCNWideResNet50

from learnt_prototypical import LearntPrototypes


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

            pred = outs.detach().cpu().numpy().argmax(axis=1)
            track_cm += confusion_matrix(labs.flatten(), pred.flatten(), labels=[0, 1])

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
        loss = criterion(outs, labs)
        # make_dot(loss).render("original", format="png")

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            pred = outs.detach().cpu().numpy().argmax(axis=1)
            acc = accuracy_score(labels.flatten(), pred.flatten())
            conf_m = confusion_matrix(labels.flatten(), pred.flatten(), labels=[0, 1])

            _sum = 0.0
            for k in range(len(conf_m)):
                _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i+1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  # " Mean Diff " + "{:.4f}".format(diff),
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / 2.0) +
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
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha value for the EMA')
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
            model = LearntPrototypes(FCNWideResNet50(train_dataset.num_classes, pretrained=True, classif=False),
                                     n_prototypes=2, embedding_dim=2560)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        # loss
        criterion = nn.CrossEntropyLoss()

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
                acc, nacc = test(test_dataloader, criterion, model, epoch)
                save_best_models(model, optimizer, args.output_path, best_records, epoch, nacc)
            scheduler.step()
    elif args.operation == 'Test':
        print('---- testing ----')
        # assert args.model_path is not None, "For inference, flag --model_path should be set."

        best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
        index = 0
        for i in range(len(best_records)):
            if best_records[index]['nacc'] < best_records[i]['nacc']:
                index = i

        print('---- data ----')
        train_dataset = DataLoader('Train', args.dataset_path, args.training_images, args.crop_size,
                                   args.stride_crop, output_path=args.output_path)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
        knn_dataset = DataLoader('KNN', args.dataset_path, args.training_images, args.crop_size,
                                 args.stride_crop, output_path=args.output_path)
        knn_dataloader = torch.utils.data.DataLoader(knn_dataset, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
        test_dataset = DataLoader('Validation', args.dataset_path, args.testing_images,
                                  args.crop_size, args.stride_crop, output_path=args.output_path)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # loss
        criterion = nn.CrossEntropyLoss()

        # network
        if args.model == 'WideResNet':
            model = FCNWideResNet50(test_dataset.num_classes, pretrained=True, classif=False)
        else:
            raise NotImplementedError("Network " + args.model + " not implemented")

        epoch = int(best_records[index]['epoch'])
        print("loading model_" + str(epoch) + '.pth')
        model.load_state_dict(torch.load(os.path.join(args.output_path, 'model_' + str(epoch) + '.pth')))
        model.cuda()

        test(knn_dataloader, test_dataloader, criterion, model, epoch)
    elif args.operation == 'Plot':
        print('---- plotting ----')
        print('---- knn data ----')
        knn_dataset = DataLoader('KNN', args.dataset_path, args.training_images, args.crop_size,
                                 args.stride_crop, output_path=args.output_path)
        knn_dataloader = torch.utils.data.DataLoader(knn_dataset, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        print('---- testing data ----')
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

        print('---- extracting feat knn ----')
        knn_feats, knn_lbs = general_feature_extractor(knn_dataloader, model)
        print('---- extracting feat test ----')
        feats, lbs = general_feature_extractor(test_dataloader, model)
        knn_lbs = knn_lbs.reshape(-1)
        lbs = lbs.reshape(-1)
        print('knn_feats', knn_feats.shape, knn_lbs.shape)
        print('feats', feats.shape, lbs.shape, np.bincount(lbs[0:100000]))
        project_data(feats[0:100000, :], lbs[0:100000], args.output_path + 'plot.png', pca_n_components=50)
    else:
        raise NotImplementedError("Operation " + args.operation + " not implemented")
