import gc
import os
import sys
import datetime
import imageio
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, jaccard_score
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
from dataloaders.data_utils import update_train_loader_entropy
from networks.FCNWideResNet50 import FCNWideResNet50
from networks.efficientnet import FCNEfficientNetB0

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
            outs_p = -outs.permute(0, 2, 3, 1).cpu().detach().numpy()

            for j in range(outs.shape[0]):
                # cur_map = cur_maps[j]
                cur_x = cur_xs[j]
                cur_y = cur_ys[j]

                prob_im[cur_x:cur_x + test_loader.dataset.crop_size,
                        cur_y:cur_y + test_loader.dataset.crop_size] += outs_p[j, :, :, 0]
                occur_im[cur_x:cur_x + test_loader.dataset.crop_size,
                         cur_y:cur_y + test_loader.dataset.crop_size] += 1

    # normalise to remove non-predicted pixels
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

    if test_loader.dataset.dataset == 'Coffee_Full' or test_loader.dataset.dataset == 'Orange':
        labs = test_loader.dataset.labels[0]
        coord = np.where(labs != 2)
        lbl = labs[coord]
        pred = prob_im_argmax[coord]
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
    jaccard = jaccard_score(lbl, pred)
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
          " Jaccard= " + "{:.4f}".format(jaccard) +
          " Tau= " + "{:.4f}".format(tau) +
          " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
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


def train_full_map(train_loader, criterion, net, epoch):
    # Setting network for evaluation mode.
    net.eval()

    prob_im = []
    occur_im = []
    for i in range(len(train_loader.dataset.labels)):
        prob_im.append(np.zeros([train_loader.dataset.labels[i].shape[0],
                                 train_loader.dataset.labels[i].shape[1]], dtype=np.float32))
        occur_im.append(np.zeros([train_loader.dataset.labels[i].shape[0],
                                  train_loader.dataset.labels[i].shape[1]], dtype=int))

    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(train_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs, cur_maps, cur_xs, cur_ys = data

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs = net(inps_c)
            outs_p = -outs.permute(0, 2, 3, 1).cpu().detach().numpy()

            for j in range(outs.shape[0]):
                cur_map = cur_maps[j]
                cur_x = cur_xs[j]
                cur_y = cur_ys[j]

                prob_im[cur_map][cur_x:cur_x + train_loader.dataset.crop_size,
                                 cur_y:cur_y + train_loader.dataset.crop_size] += outs_p[j, :, :, 0]
                occur_im[cur_map][cur_x:cur_x + train_loader.dataset.crop_size,
                                  cur_y:cur_y + train_loader.dataset.crop_size] += 1

    prob_im_argmax = []
    incorrect_classified_map = []
    for i in range(len(train_loader.dataset.labels)):
        # normalise to remove non-predicted pixels
        occur_im[i][np.where(occur_im[i] == 0)] = 1

        # prob_im_argmax = ((prob_im / occur_im.astype(float)) <= 0.999).astype(int)  # v1
        if hasattr(criterion, 'pos_margin'):
            prob_im_argmax.append(((prob_im[i] / occur_im[i].astype(float)) < criterion.pos_margin).astype(int))
        else:
            prob_im_argmax.append(((prob_im[i] / occur_im[i].astype(float)) < criterion.margin).astype(int))
        # print(prob_im_argmax[i].shape, np.bincount(prob_im_argmax[i].flatten()),
        #       train_loader.dataset.labels[i].shape, np.bincount(train_loader.dataset.labels[i].flatten()))

        incorrect_classified_map.append((prob_im_argmax[i] != train_loader.dataset.labels[i]).astype(int))

    # print(correct_classified_map[0].shape, correct_classified_map[1].shape)
    return incorrect_classified_map


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
            # pred = (-outs <= 0.999).int().detach().cpu().numpy()  # v1
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


# https://github.com/garygsw/smooth-taylor/tree/master
def calculate_smooth_taylor(test_loader, net, output_path, noise_scale=0.5, num_roots=150):
    prob_im = np.zeros([test_loader.dataset.labels[0].shape[0],
                        test_loader.dataset.labels[0].shape[1], 3], dtype=np.float64)
    occur_im = np.zeros([test_loader.dataset.labels[0].shape[0],
                         test_loader.dataset.labels[0].shape[1], 3], dtype=int)

    # Iterating over batches.
    for i, data in enumerate(test_loader):
        # Obtaining images, labels and paths for batch.
        inps, labs, cur_maps, cur_xs, cur_ys = data

        # generating white noise
        roots = torch.stack([torch.zeros_like(inps.squeeze()) for _ in range(num_roots)])
        for s in range(num_roots):
            roots[s] = inps + noise_scale * torch.randn_like(inps)

        roots_dataset = torch.utils.data.dataset.TensorDataset(roots)
        roots_data_loader = torch.utils.data.DataLoader(roots_dataset, batch_size=8, shuffle=False)

        ########################
        gradients = []
        for sample_batch in roots_data_loader:
            inputs = sample_batch[0]
            inputs = inputs.cuda()
            inputs.requires_grad = True

            # Perform the backpropagation for the explained class
            outs = net(inputs)  # batch, 1, h, w
            model.zero_grad()
            torch.sum(outs[:, 0]).backward()
            with torch.no_grad():
                gradient = inputs.grad.detach().cpu().numpy()  # retrieve the input gradients
                gradients.append(gradient)

        gradients = np.array(gradients)
        gradients = np.concatenate(gradients)

        attributions = np.mean([(inps - roots_dataset[k][0]).numpy() * gradients[k]
                                for k in range(num_roots)], axis=0)
        # print('all', gradients.shape, attributions.shape)
        ########################

        attributions = np.transpose(attributions, (0, 2, 3, 1))

        prob_im[cur_xs[0]:cur_xs[0] + test_loader.dataset.crop_size,
                cur_ys[0]:cur_ys[0] + test_loader.dataset.crop_size, :] += attributions[0, :, :, :]
        occur_im[cur_xs[0]:cur_xs[0] + test_loader.dataset.crop_size,
                 cur_ys[0]:cur_ys[0] + test_loader.dataset.crop_size, :] += 1

    # normalise to remove non-predicted pixels
    occur_im[np.where(occur_im == 0)] = 1
    saliency_map_rgb = prob_im / occur_im.astype(float)
    print(saliency_map_rgb.shape, np.min(saliency_map_rgb), np.max(saliency_map_rgb))

    # Saving
    np.save(os.path.join(output_path, 'saliency_map_rgb.npy'), saliency_map_rgb)
    imageio.imwrite(os.path.join(output_path, 'saliency_map_rgb.png'), saliency_map_rgb)

    heatmap = np.sum(saliency_map_rgb, axis=-1)
    print(heatmap.shape, np.min(heatmap), np.max(heatmap))
    np.save(os.path.join(output_path, 'saliency_map.npy'), heatmap)
    imageio.imwrite(os.path.join(output_path, 'saliency_map.png'), heatmap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True, help='Operation to be performed]',
                        choices=['Train', 'Test', 'Plot', 'Test_Full', 'Saliency_Map_Test'])
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
                        help='Model to be used.', choices=['WideResNet', 'EfficientNetB0'])
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
                        help='Miner hard samples and equalize number of samples 1:1')
    parser.add_argument('--weight_sampler', type=str2bool, default=False, help='Use weight sampler for loader?')
    parser.add_argument('--dynamic_sampler', type=str2bool, default=False, help='Dynamic sampler based on uncertainty')
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
            train_dataset = DataLoaderCoffeeFull('Full_Train', args.dataset, args.dataset_path, args.training_images,
                                                 args.crop_size, args.stride_crop, output_path=args.output_path)
            print('---- testing data ----')
            test_dataset = DataLoaderCoffeeFull('Full_Test', args.dataset, args.dataset_path, args.testing_images,
                                                args.crop_size, args.stride_crop,
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
            model = LearntPrototypes(FCNWideResNet50(train_dataset.num_classes, pretrained=True, classif=False),
                                     squared=False, n_prototypes=1, embedding_dim=2560)
        elif args.model == 'EfficientNetB0':
            model = LearntPrototypes(FCNEfficientNetB0(train_dataset.num_classes, pretrained=True, classif=False),
                                     squared=False, n_prototypes=1, embedding_dim=2096)
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
            # validation
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                # acc, nacc, f1_s, kappa, track_cm
                acc, nacc, f1_s, kappa, _ = test(test_dataloader, criterion, model, epoch)
                save_best_models(model, optimizer, args.output_path, best_records, epoch, kappa)
            # calculate patches to sample
            if args.dynamic_sampler and epoch % NEW_SAMPLE_INTERVAL == 0:
                diff_maps = train_full_map(train_dataloader, criterion, model, epoch)
                gen_classes = update_train_loader_entropy(diff_maps, train_dataset.distrib,
                                                          train_dataset.crop_size, percentage=0.10)
                train_dataloader = sample_weight_train_loader(train_dataset, gen_classes, args.batch_size)
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
            if best_records[index]['kappa'] < best_records[i]['kappa']:
                index = i

        print('---- data ----')
        if args.dataset == 'River':
            test_dataset = DataLoader('Full_test', args.dataset, args.dataset_path, args.testing_images,
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
    elif args.operation == 'Saliency_Map_Test':
        print('---- saliency map test ----')

        best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
        index = 0
        for i in range(len(best_records)):
            if best_records[index]['kappa'] < best_records[i]['kappa']:
                index = i

        print('---- data ----')
        if args.dataset == 'River':
            test_dataset = DataLoader('Full_test', args.dataset, args.dataset_path, args.testing_images,
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
        else:
            raise NotImplementedError("Dataset " + args.dataset + " not implemented")

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                      shuffle=False, num_workers=1, drop_last=False)

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

        # saliency map
        calculate_smooth_taylor(test_dataloader, model, args.output_path)
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
