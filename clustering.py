import torch


class KNN:
    def __init__(self, train_data=None, train_label=None, k=3, p=2):
        self.train_data = train_data
        self.train_label = train_label
        self.k = k
        self.p = p

    def train(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def predict(self, test_data):
        dist = torch.cdist(test_data, self.train_data, p=self.p)

        _, idxs = dist.topk(self.k, largest=False)
        labels = self.train_label[idxs]
        preds, _ = torch.mode(labels, 1)

        return preds