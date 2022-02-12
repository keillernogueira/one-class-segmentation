import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, has_miner=True, weights=[1.0, 1.0], ignore_index=-1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.has_miner = has_miner
        self.weights = torch.FloatTensor(weights).cuda()
        self.ignore_index = ignore_index

    def forward(self, data, labels):
        if len(data.shape) == 4:
            data = data.flatten()
        if len(labels.shape) == 3:
            labels = labels.flatten()

        # filtering out pixels
        coord = torch.where(labels != self.ignore_index)
        labels = labels[coord]
        data = data[coord]

        if self.has_miner:
            data, labels, _ = self.miner(data, labels)

        poss = torch.gather(data.flatten(), 0, labels.flatten().nonzero().squeeze())
        # print('pos ', torch.min(poss).data.item(), torch.mean(poss).data.item(), torch.max(poss).data.item())
        negs = torch.gather(data.flatten(), 0, (1 - labels).flatten().nonzero().squeeze())
        # print('negs', torch.min(negs).data.item(), torch.mean(negs).data.item(), torch.max(negs).data.item())

        p = torch.mean(labels * torch.pow(data, 2))
        n = torch.mean((1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        a = torch.mean(labels * torch.pow(data, 2) +
                       (1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        print('loss', torch.mean(poss).data.item(), torch.mean(negs).data.item(), p, n, a)
        loss_contrastive = torch.mean(self.weights[1] * labels * torch.pow(data, 2) +
                                      self.weights[0] * (1 - labels) *
                                      torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        return loss_contrastive

    def miner(self, data, labels):
        all_pos_values = data[labels.bool()]
        all_neg_values = data[(1 - labels).bool()]

        # get all **hard** samples of the negative class with dist <= margin
        neg_hard = all_neg_values[all_neg_values < self.margin]  # I used '<=' in first version

        # print('1', neg_hard.shape, all_neg_values.shape, all_pos_values.shape)

        if neg_hard.shape[0] != 0:
            if all_pos_values.shape >= neg_hard.shape:
                # get the same number of **hard** samples of the positive class
                pos_hard, _ = torch.topk(all_pos_values, neg_hard.shape[0], largest=True)
            else:
                pos_hard = all_pos_values  # get all positive samples
                # with the line below commented out, it was called v2
                # neg_hard, _ = torch.topk(neg_hard, pos_hard.shape[0], largest=False)  # this is v1
        else:
            total = torch.bincount(labels)[0] + torch.bincount(labels)[1]
            weights = torch.FloatTensor([1.0 + torch.true_divide(torch.bincount(labels)[1], total),
                                         1.0 + torch.true_divide(torch.bincount(labels)[0], total)]).cuda()
            # print('----- 1 proportion ', torch.bincount(labels), total, weights)
            return data, labels, weights
            # uncomment lines below to balance dataset
            # pos_hard = all_pos_values  # get all positive samples
            # neg_hard, _ = torch.topk(all_neg_values, pos_hard.shape[0], largest=False)

        neg_labels = torch.zeros(neg_hard.shape, device='cuda:0')
        pos_labels = torch.ones(pos_hard.shape, device='cuda:0')

        # print('4', neg_labels.shape[0], pos_labels.shape[0], neg_hard.shape[0], pos_hard.shape[0])
        #       torch.min(all_neg_values), torch.max(all_neg_values),
        #       torch.min(all_pos_values), torch.max(all_pos_values),
        #       torch.min(neg_hard), torch.max(neg_hard), torch.min(pos_hard), torch.max(pos_hard))
        # print('----------------------------------------------------------------------------------------------------')
        total = neg_labels.shape[0] + pos_labels.shape[0]
        weights = torch.FloatTensor([1.0+torch.true_divide(pos_labels.shape[0], total),
                                     1.0+torch.true_divide(neg_labels.shape[0], total)]).cuda()
        # print('----- 2 proportion ', neg_labels.shape, pos_labels.shape, total, weights)

        return torch.cat([neg_hard, pos_hard]), torch.cat([neg_labels, pos_labels]), weights
