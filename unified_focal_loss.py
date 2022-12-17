import torch
import torch.nn as nn
import torch.nn.functional as F


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/16
class FocalLossV2(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLossV2, self).__init__()
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/tversky.html
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Focal-Tversky-Loss
class FocalTverskyLoss(nn.Module):
    def __init__(self, delta=0.7, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.eps = 1e-6

    def one_hot(self, labels, num_classes, dtype=None):
        if not torch.is_tensor(labels):
            raise TypeError("Input labels type is not a torch.Tensor. Got {}".format(type(labels)))
        if not len(labels.shape) == 3:
            raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}".format(labels.shape))
        if not labels.dtype == torch.int64:
            raise ValueError("labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype))
        if num_classes < 1:
            raise ValueError("The number of classes must be bigger than one. Got: {}".format(num_classes))
        batch_size, height, width = labels.shape
        one_hot = torch.zeros(batch_size, num_classes, height, width, dtype=dtype).cuda()
        return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)  # + eps

    def forward(self, inputs, targets):
        input_soft = F.softmax(inputs, dim=1)

        # create the labels one hot tensor
        target_one_hot = self.one_hot(targets, num_classes=inputs.shape[1], dtype=inputs.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        fps = torch.sum(input_soft * (1. - target_one_hot), dims)
        fns = torch.sum((1. - input_soft) * target_one_hot, dims)

        denominator = intersection + self.delta * fps + (1 - self.delta) * fns
        tversky_loss = (intersection + self.eps) / (denominator + self.eps)
        return torch.mean((1. - tversky_loss) ** self.gamma)


# https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py
class UnifiedFocalLoss(nn.Module):
    def __init__(self, internal_weight=None, weight=None, delta=0.6, gamma=2):
        super(UnifiedFocalLoss, self).__init__()
        self.gamma = gamma
        self.delta = delta
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights
        self.internal_weight = internal_weight
        self.focal = FocalLossV2(weight, gamma)
        self.tversky = FocalTverskyLoss(delta, gamma)

    def forward(self, inputs, targets):
        focal = self.focal(inputs, targets)
        tversky = self.tversky(inputs, targets)
        if self.internal_weight is not None:
            return (self.internal_weight * tversky) + ((1 - self.internal_weight) * focal)
        else:
            return tversky + focal

