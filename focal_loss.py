import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
# https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        print('inside', self.gamma, self.alpha)

    def forward(self, inputs, targets):
        if len(inputs.shape) == 3:
            inputs = inputs.flatten()
        if len(targets.shape) == 3:
            targets = targets.flatten()

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)  # cross entropy without log
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        if len(inputs.shape) == 4:
            inputs = inputs.flatten()
        if len(targets.shape) == 3:
            targets = targets.flatten()
        p_t = F.softmax(inputs)  # probabilities
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        # p_t = p * targets + (1 - p) * (1 - targets)  # cross entropy without log
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        return loss.mean()


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
