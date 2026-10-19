import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(emb1.float(), emb2.t().float(), beta=1, alpha=-2)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


class TripletLoss_ADP(nn.Module):
    def __init__(self, alpha=1, gamma=1, square=1):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, targets, normalize_feature=False):
        if len(targets.shape) > 1:
            targets = torch.max(targets, 1)[1]

        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap * self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an * self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)
        # print(inputs)
        # print(dist_mat)
        # print(-dist_an * self.alpha)
        # print(is_neg)
        # print(dist_an)
        # print(weights_an)
        # print(dist_an * weights_an)

        # squared difference
        if self.square == 0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            diff = closest_negative - furthest_positive
            # print(closest_negative)
            # print(furthest_positive)
            # print(diff)
            diff = torch.clamp_max(diff, max=88)
            diff = torch.clamp_min(diff, min=-88)
            # print(diff)
            loss = self.ranking_loss(self.gamma * diff, y)
            # print(loss)
            # exit(0)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow = torch.clamp_max(diff_pow, max=88)

            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)

            loss = self.ranking_loss(diff_pow, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


class KLDivLoss(nn.Module):
    def __init__(self, T=3):
        super(KLDivLoss, self).__init__()
        self.T = T

    def forward(self, pred, label):

        predict = F.log_softmax(pred / self.T, dim=1)
        target_data = F.softmax(label / self.T, dim=1)
        target_data = target_data + 10 ** (-7)
        target = target_data.data
        loss = self.T * self.T * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        return loss
