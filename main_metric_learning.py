import gc
import sys
import datetime

import torch
from torch import optim
from torch.autograd import Variable

from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.reducers import MultipleReducers, ThresholdReducer, AvgNonZeroReducer

from dataloaders.dataloader import DataLoader
from config import *
from utils import *
from networks.FCNWideResNet50 import FCNWideResNet50
from feat_ext import general_feature_extractor


def test(test_loader, val_dataloader, net, epoch):
    # Setting network for evaluation mode.
    net.eval()

    selected_val_samples = None
    selected_cal_labs = None
    acc_calc = AccuracyCalculator(return_per_class=True, k='max_bin_count')

    average_acc = np.zeros(2)
    counter = 0

    with torch.no_grad():
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
            break

        print(selected_cal_labs.shape, selected_cal_labs.shape)

        # Iterating over batches.
        for i, data in enumerate(test_loader):

            # Obtaining images, labels and paths for batch.
            inps, labs = data[0], data[1]

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            _, fv2, fv4 = net(inps_c)  # output here are the logits and 2 feature vectors

            # Concatenating features
            feat_flat = torch.cat([fv2, fv4], 1)
            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(-1, feat_flat.size(1))

            all = acc_calc.get_accuracy(feat_flat.cpu().detach().numpy(),
                                        selected_val_samples.cpu().detach().numpy(),
                                        labs.view(-1).cpu().detach().numpy(),
                                        selected_cal_labs.view(-1).cpu().detach().numpy(),
                                        embeddings_come_from_same_source=False)
            average_acc += all['mean_average_precision']
            counter += 1

        average_acc /= counter
        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(np.average(average_acc)) +
              " Confusion Matrix= " + np.array_str(average_acc).replace("\n", "")
              # " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
              # " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
              )

        sys.stdout.flush()

    return np.average(average_acc)


def train(train_loader, net, tl_criterion, optimizer, epoch):
    # Setting network for training mode.
    net.train()

    # Average Meter for batch loss.
    train_loss = list()

    acc_calc = AccuracyCalculator(return_per_class=True, k='max_bin_count')

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
        _, fv2, fv4 = net(inps)  # output here are the logits and 2 feature vectors

        # Concatenating features
        feat_flat = torch.cat([fv2, fv4], 1)
        feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(-1, feat_flat.size(1))

        # Triplet loss
        # hard_pairs = miner(feat_flat, labels.view(-1))
        loss = tl_criterion(feat_flat, labels.view(-1))  # , hard_pairs)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            # Obtaining predictions.
            all = acc_calc.get_accuracy(feat_flat.cpu().detach().numpy(), feat_flat.cpu().detach().numpy(),
                                        labels.view(-1).cpu().detach().numpy(), labels.view(-1).cpu().detach().numpy(),
                                        embeddings_come_from_same_source=True)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i+1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(np.average(all['mean_average_precision'])) +
                  " Mean Average Precision= [" + ','.join(str(e) for e in all['mean_average_precision']) + "]"
                  # " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
                  # " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )
            # sys.stdout.flush()

    gc.collect()


# https://kevinmusgrave.github.io/pytorch-metric-learning/
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
    parser.add_argument('--reducer', type=str2bool, default=False, help='Custom reducer')
    args = parser.parse_args()
    print(args)

    # data loaders
    if args.operation == 'Train':
        print('---- training data ----')
        train_dataset = DataLoader('Train', args.dataset_path, args.training_images, args.crop_size,
                                   args.stride_crop, output_path=args.output_path)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=NUM_WORKERS, drop_last=False)

        print('---- val data ----')
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
        # miner = miners.MultiSimilarityMiner()
        # tl_criterion = losses.TripletMarginLoss(margin=args.margin)
        if args.reducer is True:
            reducer_dict = {"pos_loss": ThresholdReducer(0.1), "neg_loss": AvgNonZeroReducer()}
            reducer = MultipleReducers(reducer_dict)
            tl_criterion = losses.ContrastiveLoss(neg_margin=args.margin, reducer=reducer)
            print("reducer")
        else:
            tl_criterion = losses.ContrastiveLoss(neg_margin=args.margin)

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
            train(train_dataloader, model, tl_criterion, optimizer, epoch)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                nacc = test(test_dataloader, knn_dataloader, model, epoch)
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
    elif args.operation == 'Plot':
        print('---- plotting ----')
        print('---- knn train data ----')
        knn_dataset = DataLoader('KNN', args.dataset_path, args.training_images, args.crop_size,
                                 args.stride_crop, output_path=args.output_path)
        knn_dataloader = torch.utils.data.DataLoader(knn_dataset, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        print('---- knn test data ----')
        # DataLoader('KNN', args.dataset_path, args.testing_images,
        test_dataset = DataLoader('KNN', args.dataset_path, args.testing_images,
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
        knn_lbs = knn_lbs.reshape(-1)
        print('knn_feats', knn_feats.shape, knn_lbs.shape)
        project_data(knn_feats, knn_lbs, args.output_path + 'train_plot.png', pca_n_components=50)

        print('---- extracting feat test ----')
        feats, lbs = general_feature_extractor(test_dataloader, model)
        lbs = lbs.reshape(-1)
        print('feats', feats.shape, lbs.shape, np.bincount(lbs[0:100000]))
        project_data(feats, lbs, args.output_path + 'test_plot.png', pca_n_components=50)
        # project_data(feats[0:100000, :], lbs[0:100000], args.output_path + 'plot.png', pca_n_components=50)
    else:
        raise NotImplementedError("Operation " + args.operation + " not implemented")
