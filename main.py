import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

from dataloader import DataLoader
from config import *
from utils import *
from network import FCNWideResNet50

from train import train
from test import test_per_patch


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True, help='Operation to be performed]',
                        choices=['Train', 'Test', 'Plot'])
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save outcomes (such as images and trained models) of the algorithm.')
    parser.add_argument('--train_strategy', type=str, required=True,
                        choices=['original', 'detach', 'track_mean', 'hard_triplet', 'std_hard_triplet'],
                        help='Possible strategies to train model. Options: '
                             'original: no real track mean and not detach, '
                             'detach: no real track mean but detach graph, '
                             'track_mean: real track mean, '
                             'hard_triplet: hard triplet mining, '
                             'std_hard_triplet: standard hard triplet - batch size = 2')
    parser.add_argument('--test_strategy', type=str, required=True, choices=['track_mean', 'knn'],
                        help='Possible strategies to test model. Options: '
                             'track_mean: use track mean to predict, '
                             'knn: use knn to predict')

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
    parser.add_argument('--k', type=int, default=11, help='k for the kNN')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha value for the EMA')
    parser.add_argument('--margin', type=float, default=1.0, help='Triplet loss margin')
    args = parser.parse_args()
    print(args)

    # data loaders
    if args.operation == 'Train':
        print('---- training data ----')
        train_dataset = DataLoader('Train', args.dataset_path, args.training_images, args.crop_size,
                                   args.stride_crop, output_path=args.output_path)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=2 if args.train_strategy == 'std_hard_triplet' else args.batch_size,
                                                       shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
        knn_dataloader = None
        if args.test_strategy == 'knn':
            print('---- knn data ----')
            knn_dataset = DataLoader('KNN', args.dataset_path, args.training_images, args.crop_size,
                                     args.stride_crop, mean=train_dataset.mean, std=train_dataset.std)
            knn_dataloader = torch.utils.data.DataLoader(knn_dataset, batch_size=args.batch_size,
                                                         shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
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

        # loss
        ce_criterion = nn.CrossEntropyLoss().cuda()
        tl_criterion = nn.TripletMarginLoss(margin=args.margin, p=2).cuda()  # TODO check cuda ??

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                               betas=(0.9, 0.99))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

        # load model and readjust scheduler
        curr_epoch = 1
        best_records = []
        if args.model_path is not None:
            print('Loading model ' + args.model_path)
            best_records = np.load(os.path.join(args.output_path, 'best_records.npy'))
            model.load_state_dict(torch.load(args.model_path))
            # optimizer.load_state_dict(torch.load(args.model_path.replace("model", "opt")))
            curr_epoch += int(os.path.basename(args.model_path)[:-4].split('_')[-1])
            for i in range(curr_epoch):
                scheduler.step()
        model.cuda()

        print('---- training ----')
        for epoch in range(curr_epoch, args.epoch_num + 1):
            track_mean = train(train_dataloader, model, tl_criterion, optimizer, epoch, args.alpha, args.margin,
                               args.train_strategy, args.test_strategy)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                acc, nacc = test_per_patch(test_dataloader, model, epoch,
                                           track_mean=track_mean, knn_dataloader=knn_dataloader, k=args.k)
                save_best_models(model, optimizer, args.output_path, best_records, epoch, nacc, track_mean=track_mean)
            scheduler.step()
    elif args.operation == 'Test':
        print('---- testing ----')
        assert args.model_path is not None, "For inference, flag --model_path should be set."

        best_records = np.load(os.path.join(args.output_path, 'best_records.npy'))
        index = 0
        for i in range(len(best_records)):
            if best_records[index]['nacc'] < best_records[i]['nacc']:
                index = i
        track_mean = best_records[index]['track_mean']

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

        test_per_patch(test_dataloader, model, epoch, track_mean=track_mean)
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
