import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, weights=[1.0, 1.0], ignore_index=-1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.weights = torch.FloatTensor(weights).cuda()
        self.ignore_index = ignore_index

    def forward(self, data, embeddings, labels):
        if len(data.shape) == 4:
            data = data.flatten()
        if len(labels.shape) == 3:
            labels = labels.flatten()

        # filtering out pixels
        # print('check 0', labels.shape, data.shape, embeddings.shape)
        labels = labels[labels != self.ignore_index]
        data = data[labels != self.ignore_index]
        embeddings = embeddings[labels != self.ignore_index, :]
        # print('check 1', labels.shape, data.shape, embeddings.shape)

        pos_samples = data[labels.bool()]
        print('pos ', pos_samples.shape, labels.shape, torch.bincount(labels), torch.min(pos_samples).data.item(),
              torch.mean(pos_samples).data.item(), torch.max(pos_samples).data.item())

        hard_neg_embs = self.miner_hard_neg(data, embeddings, labels)
        if hard_neg_embs.shape[0] == 0:
            loss_contrastive = torch.mean(self.weights[1] * torch.pow(pos_samples, 2))
        else:
            neg_dist = torch.cdist(hard_neg_embs, embeddings[labels.bool()], p=2)
            # print('check 2', hard_neg_embs.shape, embeddings[labels.bool()].shape, neg_dist.shape)

            print('losses contrib', torch.mean(self.weights[1] * torch.pow(pos_samples, 2)),
                  torch.mean(self.weights[0] * torch.pow(torch.clamp(self.margin - neg_dist.flatten(), min=0.0), 2)))
            loss_contrastive = torch.mean(self.weights[1] * torch.pow(pos_samples, 2)) + \
                               torch.mean(self.weights[0] * torch.pow(torch.clamp(self.margin - neg_dist.flatten(), min=0.0), 2))
        return loss_contrastive

    def miner_hard_neg(self, data, embeddings, labels):
        all_neg_values = data[(1 - labels).bool()]
        all_neg_embs = embeddings[(1 - labels).bool()]

        # get all **hard** samples of the negative class with dist < margin
        neg_hard = all_neg_embs[all_neg_values < self.margin]
        print('hard samples qty ', neg_hard.shape)

        if neg_hard.shape[0] > 5000:
            neg_hard = neg_hard[0:5000, :]

        return neg_hard
