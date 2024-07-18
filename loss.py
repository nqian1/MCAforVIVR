import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd.function import Function
from torch.autograd import Variable

def hardming(dist,targets):
    n = dist.size(0)
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    dist_ap, dist_an = [], []
    for i in range(n):
        dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    return dist_ap, dist_an

# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v

    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt() #12
    # print(dist_mtx)
    return dist_mtx

def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis=1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis=1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

def find_hard(dist_mat, targets):
    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
    is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)
    furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
    closest_negative = torch.sum(dist_an * weights_an, dim=1)

    return furthest_positive, closest_negative

def find_hardv2(dist_mat, targets):
    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
    is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)
    furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
    closest_negative = torch.sum(dist_an * weights_an, dim=1)

    dist_mat_weight = dist_ap * weights_ap + dist_an * weights_an

    return furthest_positive, closest_negative, dist_mat_weight

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        # print(num_classes)
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        # print(targets.size())
        # print(targets)
        if torch.any(torch.isnan(inputs)):
            print('fuck you')
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        # self.margin = margin
        self.ranking_loss = nn.SoftMarginLoss()
        # self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """

        n = inputs.size(0)
        # rgb = inputs[0:n // 2, :]
        # nir = inputs[n // 2:, :]
        # classes = torch.unique(targets)
        # num_classes = len(classes)
        # instances = rgb.size(0) // num_classes

        dist = pdist_torch(inputs,inputs)
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = ~pos_mask


        dist_ap = torch.max(dist[pos_mask].view(n,-1),dim=1)[0]
        dist_an = torch.min(dist[neg_mask].view(n,-1),dim=1)[0]
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an - dist_ap, y)
        # loss = self.ranking_loss(dist_an, dist_ap, y) # for margin, stronger
        return loss

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self,margin=0.0):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()



    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)

        n = inputs.size(0)
        rgb = inputs[0:n // 2, :]
        nir = inputs[n // 2:, :]
        classes = torch.unique(targets)
        num_classes = len(classes)
        instances = rgb.size(0) // num_classes
        rgb = rgb.view(num_classes, instances, -1)
        nir = nir.view(num_classes, instances, -1)



        dist = pdist_torch(inputs, inputs)
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = ~pos_mask

        dist_ap = dist[pos_mask].view(n,-1)
        dist_an = dist[neg_mask].view(n,-1)

        weights_ap = F.softmax(dist_ap,dim=1)
        weights_an = F.softmax(-dist_an,dim=1)

        dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
        dist_an = torch.sum(dist_an * weights_an, dim=1)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an - dist_ap, y)

        # compute accuracy
        # correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss

class MSE(nn.Module):
    def __init__(self, ):
        super(MSE, self).__init__()
        self.margin = 1e-3
        # self.tau = 2

    def forward(self, nir,rgb,label):

        uid=torch.unique(label)
        num_classes=len(uid)
        # print('num_classes')
        # print(num_classes)
        instances = rgb.size(0) // num_classes
        # print('instances.shape')
        # print(instances)
        rgb = rgb.view(num_classes, instances, -1).sum(1)
        nir = nir.view(num_classes, instances, -1).sum(1)
        # print('rgb')
        # print(rgb.shape)
        # print('nir')
        # print(nir.shape)

        rgb = F.normalize(rgb, p=2, dim=1)
        nir = F.normalize(nir, p=2, dim=1)
        # print('rgbn')
        # print(rgb.shape)
        # print('nirn')
        # print(nir.shape)

        diff = (nir-rgb)*(nir-rgb)

        diff = diff.sum(dim=1)

        diff = torch.clamp(diff,min=1e-12).sqrt()

        loss  = diff.mean(0)

        return loss

