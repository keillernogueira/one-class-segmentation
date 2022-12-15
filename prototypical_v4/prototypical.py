import torch
import torch.nn as nn
from torch.nn import functional as F

from sklearn import metrics


class PrototypicalLoss(nn.Module):
    def __init__(self):
        super(PrototypicalLoss, self).__init__()
        self.protos = None

        # this is for prediction
        self.average_pos = []
        self.average_neg = []
        self.num_pos = []
        self.num_neg = []

        self.logits = None
        self.pred = None

    def forward(self, embeddings, labels):
        dists = distance_logits(self.protos, embeddings)  # [B * H * W, 2]
        self.logits = F.log_softmax(-dists, dim=1)
        _, self.pred = self.logits.max(1)

        return -self.logits.gather(1, labels.view(-1, 1)).squeeze().view(-1).mean()

    def update_prototypes(self, embeddings, labels):
        neg_proto = embeddings[labels == 0, :].mean(dim=0)
        pos_proto = embeddings[labels == 1, :].mean(dim=0)

        self.protos = torch.stack([neg_proto, pos_proto])

    def predict(self, data):
        dists = distance_logits(self.protos, data)
        log_p_y = F.log_softmax(-dists, dim=1)
        _, y_hat = log_p_y.max(1)

        return y_hat

    def update_averages(self, embeddings, labels):
        self.average_neg.append(embeddings[labels == 0, :].mean(dim=0))
        self.average_pos.append(embeddings[labels == 1, :].mean(dim=0))
        self.num_neg.append(embeddings[labels == 0, :].shape[0])
        self.num_pos.append(embeddings[labels == 1, :].shape[0])

    def update_average_prototypes(self):
        neg = 0
        pos = 0
        for idx in range(len(self.average_neg)):
            neg += self.average_neg[idx] * self.num_neg[idx]
            pos += self.average_pos[idx] * self.num_pos[idx]
        neg_proto = neg / sum(self.num_neg)
        pos_proto = pos / sum(self.num_pos)

        self.protos = torch.stack([neg_proto, pos_proto])


def distance_logits(sup, qry):
    n_samples = qry.size(0)
    d_feat = qry.size(1)
    m_prototypes = sup.size(0)

    qry = qry.unsqueeze(1).expand(n_samples, m_prototypes, d_feat)
    sup = sup.unsqueeze(0).expand(n_samples, m_prototypes, d_feat)

    # logits = -((qry_exp - sup_exp) ** 2).sum(dim=1)  # hugo/pedro implementation
    logits = torch.pow(qry - sup, 2).sum(2)  # https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_loss.py

    return logits


def accuracy(lab, prd):
    # Obtaining class from prediction.
    prd = prd.argmax(1)

    # Tensor to ndarray.
    lab_np = lab.view(-1).detach().cpu().numpy()
    prd_np = prd.view(-1).detach().cpu().numpy()

    # Computing metric and returning.
    metric_val = metrics.jaccard_score(lab_np, prd_np)

    return metric_val


# deprecated
class ProtoPred:
    def __init__(self, val_data=None, val_label=None):
        self.val_data = val_data
        self.val_label = val_label
        neg_feat = val_data[val_label == 0, :]
        pos_feat = val_data[val_label == 1, :]
        proto_neg = neg_feat.mean(dim=0)
        proto_pos = pos_feat.mean(dim=0)
        self.proto_sup = torch.stack([proto_neg, proto_pos])


