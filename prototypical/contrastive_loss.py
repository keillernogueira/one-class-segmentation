import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, has_miner=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.has_miner = has_miner

    def forward(self, data, labels):
        if len(data.shape) == 4:
            data = data.flatten()
        if len(labels.shape) == 3:
            labels = labels.flatten()

        if self.has_miner:
            data, labels = self.miner(data, labels)

        # poss = torch.gather(data.flatten(), 0, labels.flatten().nonzero().squeeze())
        # negs = torch.gather(data.flatten(), 0, (1 - labels).flatten().nonzero().squeeze())
        # print('1', torch.min(negs).data, torch.max(negs), torch.min(poss), torch.max(poss))
        # print('pos ', torch.mean(labels * torch.pow(data, 2)))
        # print('neg ', torch.mean((1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2)))
        loss_contrastive = torch.mean(labels * torch.pow(data, 2) +
                                      (1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        return loss_contrastive

    def miner(self, data, labels):
        all_pos_values = data[labels.bool()]
        all_neg_values = data[(1 - labels).bool()]

        # get all **hard** samples of the negative class with dist <= margin
        neg_hard = all_neg_values[all_neg_values <= self.margin]

        # print('1', neg_hard.shape, all_neg_values.shape, all_pos_values.shape)

        if neg_hard.shape[0] != 0:
            if all_pos_values.shape >= neg_hard.shape:
                # get the same number of **hard** samples of the positive class
                pos_hard, _ = torch.topk(all_pos_values, neg_hard.shape[0], largest=True)
            else:
                pos_hard = all_pos_values  # get all positive samples
                neg_hard, _ = torch.topk(neg_hard, pos_hard.shape[0], largest=False)
        else:
            return data, labels
            # uncomment lines below to balance dataset
            # pos_hard = all_pos_values  # get all positive samples
            # neg_hard, _ = torch.topk(all_neg_values, pos_hard.shape[0], largest=False)

        neg_labels = torch.zeros(neg_hard.shape, device='cuda:0')
        pos_labels = torch.ones(pos_hard.shape, device='cuda:0')

        # print('4', neg_labels.shape[0], pos_labels.shape[0], neg_hard.shape[0], pos_hard.shape[0],
        #       torch.min(all_neg_values), torch.max(all_neg_values),
        #       torch.min(all_pos_values), torch.max(all_pos_values),
        #       torch.min(neg_hard), torch.max(neg_hard), torch.min(pos_hard), torch.max(pos_hard))
        # print('----------------------------------------------------------------------------------------------------')

        return torch.cat([neg_hard, pos_hard]), torch.cat([neg_labels, pos_labels])
