import numpy as np

import torch
from torch.autograd import Variable


def general_feature_extractor(test_loader, net):
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
