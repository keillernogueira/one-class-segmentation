import torch
import torch.nn as nn
import torch.nn.functional as F


# https://www.sciencedirect.com/science/article/pii/S0925231221011310
class DualFocalLoss(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1, rho=1):
        super(DualFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho

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

        # predics = torch.gather(input_soft, dim=1, index=targets[:, None, :, :])
        # predics = torch.max(input_soft, dim=1)
        # create the labels one hot tensor
        target_one_hot = self.one_hot(targets, num_classes=inputs.shape[1], dtype=inputs.dtype)

        dims = (1, 2, 3)
        first_term = torch.sum(torch.log(input_soft + self.eps) * target_one_hot, dims)
        second_term = self.beta * torch.sum((1. - target_one_hot) * torch.log(self.rho - input_soft + self.eps), dims)
        third_term = self.alpha * torch.sum(torch.abs(target_one_hot - input_soft) ** self.gamma, dims)
        # print('test', torch.min(input_soft), torch.max(input_soft),
        #       torch.min(self.rho - input_soft), torch.max(self.rho - input_soft))
        # print('1', first_term)
        # print('2', second_term)
        # print('3', third_term)
        # print('loss', -(first_term + second_term + third_term).mean())

        return -(first_term + second_term + third_term).mean()
