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
        b, c, h, w = data.shape
        # change axes to make predictions the first dimension
        data = data.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)
        labels = labels.flatten()
        # print('1', data.shape, labels.shape, data.type(), labels.type())

        # filtering out pixels
        coord = torch.where(labels != self.ignore_index)
        labels = labels[coord]
        data = data[coord]
        # print('filter2', data.shape, labels.shape)

        pos_data = data[labels.bool(), :]
        neg_data = data[(1 - labels).bool(), :]
        # print('2', pos_data.shape, neg_data.shape)

        pos_data_min, _ = pos_data.min(1)  # for each sample, get the minimum value == the closest prototype
        neg_data = neg_data.flatten()
        # print('3', pos_data_min.shape, neg_data.shape)

        data = torch.cat([pos_data_min, neg_data])
        labels = torch.cat([torch.ones(pos_data_min.shape, dtype=torch.long, device='cuda:0'),
                            torch.zeros(neg_data.shape, dtype=torch.long, device='cuda:0')])
        # print('4', data.shape, labels.shape, data.type(), labels.type())

        if self.has_miner:
            data, labels, self.weights = self.miner_v3(data, labels)

        # poss = torch.gather(data.flatten(), 0, labels.flatten().nonzero().squeeze())
        # print('pos ', torch.min(poss).data.item(), torch.mean(poss).data.item(), torch.max(poss).data.item())
        # negs = torch.gather(data.flatten(), 0, (1 - labels).flatten().nonzero().squeeze())
        # print('negs', torch.min(negs).data.item(), torch.mean(negs).data.item(), torch.max(negs).data.item())

        # p = torch.mean(labels * torch.pow(data, 2))
        # n = torch.mean((1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        # a = torch.mean(labels * torch.pow(data, 2) +
        #                (1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        # print('loss', torch.mean(poss).data.item(), torch.mean(negs).data.item(), p.data.item(),
        #       n.data.item(), a.data.item())
        loss_contrastive = torch.mean(self.weights[1] * labels * torch.pow(data, 2) +
                                      self.weights[0] * (1 - labels) *
                                      torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        return loss_contrastive

    def miner_v3(self, data, labels):
        all_pos_values = data[labels.bool()]
        all_neg_values = data[(1 - labels).bool()]

        # get all **hard** samples of the negative class with dist < margin
        neg_hard = all_neg_values[all_neg_values < self.margin]

        if neg_hard.shape[0] == 0:
            total = torch.bincount(labels)[0] + torch.bincount(labels)[1]
            weights = torch.FloatTensor([1.0 + torch.true_divide(torch.bincount(labels)[1], total),
                                         1.0 + torch.true_divide(torch.bincount(labels)[0], total)]).cuda()
            return data, labels, weights

        neg_labels = torch.zeros(neg_hard.shape, device='cuda:0')
        pos_labels = torch.ones(all_pos_values.shape, device='cuda:0')

        total = neg_labels.shape[0] + pos_labels.shape[0]
        weights = torch.FloatTensor([1.0+torch.true_divide(pos_labels.shape[0], total),
                                     1.0+torch.true_divide(neg_labels.shape[0], total)]).cuda()

        return torch.cat([neg_hard, all_pos_values]), torch.cat([neg_labels, pos_labels]), weights
