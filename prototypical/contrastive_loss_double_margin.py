import torch
import torch.nn as nn


class ContrastiveLossDoubleMargin(nn.Module):
    def __init__(self, margin=1.0, pos_margin=0.7, has_miner=True, weights=[1.0, 1.0], ignore_index=-1):
        super(ContrastiveLossDoubleMargin, self).__init__()
        self.margin = margin
        self.pos_margin = pos_margin

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
            data, labels = self.miner(data, labels)

        # poss = torch.gather(data.flatten(), 0, labels.flatten().nonzero().squeeze())
        # negs = torch.gather(data.flatten(), 0, (1 - labels).flatten().nonzero().squeeze())
        # print('negs', torch.min(negs).data.item(), torch.mean(negs).data.item(), torch.max(negs).data.item())
        # print('pos ', torch.min(poss).data.item(), torch.mean(poss).data.item(), torch.max(poss).data.item())

        # p = torch.mean(labels * torch.pow(torch.clamp(data - self.pos_margin, min=0.0), 2))
        # n = torch.mean((1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        # a = torch.mean(labels * torch.pow(torch.clamp(data - self.pos_margin, min=0.0), 2) +
        #                (1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        # print('loss', p, n, a)
        loss_contrastive = torch.mean(self.weights[1] *
                                      labels * torch.pow(torch.clamp(data - self.pos_margin, min=0.0), 2) +
                                      self.weights[0] *
                                      (1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        return loss_contrastive

    def miner(self, data, labels):
        all_pos_values = data[labels.bool()]
        all_neg_values = data[(1 - labels).bool()]

        # get all **hard** samples of the negative class with dist < margin
        neg_hard = all_neg_values[all_neg_values < self.margin]
        # get all **hard** samples of the negative class with dist <= margin
        pos_hard = all_pos_values[all_pos_values > self.pos_margin]

        # print('hards shape', neg_hard.shape, pos_hard.shape)

        # if neg_hard.shape[0] != 0 and pos_hard.shape[0] != 0:
        #     if pos_hard.shape >= neg_hard.shape:
        #         # get the same number of **hard** samples of the positive class
        #         pos_hard, _ = torch.topk(all_pos_values, neg_hard.shape[0], largest=True)
        #     else:
        #         neg_hard, _ = torch.topk(neg_hard, pos_hard.shape[0], largest=False)

        # print('final hards shape', neg_hard.shape, pos_hard.shape)
        neg_labels = torch.zeros(neg_hard.shape, device='cuda:0')
        pos_labels = torch.ones(pos_hard.shape, device='cuda:0')

        return torch.cat([neg_hard, pos_hard]), torch.cat([neg_labels, pos_labels])
