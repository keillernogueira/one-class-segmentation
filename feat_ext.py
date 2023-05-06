import numpy as np

import torch
from torch.autograd import Variable


def general_feature_extractor(test_loader, net, amount=10000, percentage=0.01):
    # Setting network for evaluation mode.
    net.eval()

    all_labels = None
    all_feats = None
    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):

            # Obtaining images, labels and paths for batch.
            inps, labs = data[0], data[1]

            # ensure there will be water classes
            if len(torch.bincount(labs.view(-1))) != 2:
                continue

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            _, embeddings = net(inps_c)
            # print('embeddings 1 ', embeddings.shape)

            b, c, h, w = embeddings.shape
            embeddings = embeddings.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c).cpu().detach().numpy()

            np_labs = labs.cpu().detach().numpy().reshape(-1)
            # print('embeddings 2 ', embeddings.shape, np_labs.shape)

            selected_index = np.random.choice(len(np_labs), int(len(np_labs)*percentage))

            if all_labels is None:
                all_labels = np_labs[selected_index]
                all_feats = embeddings[selected_index]
            else:
                all_labels = np.concatenate((all_labels, np_labs[selected_index]))
                all_feats = np.concatenate((all_feats, embeddings[selected_index]))

            print('final', all_labels.shape, all_feats.shape, np.bincount(all_labels))
            if len(all_labels) > amount:
                break

    return all_feats, all_labels
