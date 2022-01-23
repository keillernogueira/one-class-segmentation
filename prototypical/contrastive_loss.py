import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, data, labels):
        if len(data.shape) == 4:
            data = data.flatten()
        if len(labels.shape) == 3:
            labels = labels.flatten()

        # poss = torch.gather(data.flatten(), 0, labels.flatten().nonzero().squeeze())
        # negs = torch.gather(data.flatten(), 0, (1 - labels).flatten().nonzero().squeeze())
        # print('1', torch.min(negs).data, torch.max(negs), torch.min(poss), torch.max(poss))
        # print('pos ', torch.mean(labels * torch.pow(data, 2)))
        # print('neg ', torch.mean((1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2)))
        loss_contrastive = torch.mean(labels * torch.pow(data, 2) +
                                      (1 - labels) * torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        return loss_contrastive
