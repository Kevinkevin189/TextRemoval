import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16 as vgg
import numpy as np
import pickle
import os.path as path


# for this loss part ,it consists of 6 individual loss parts
# tensors which take parts in calculation are:
# output(abbr. out)
# ground-truth(abbr. gt)
# mask(abbr. mask,stands for seg-gt)
# composited output(abbr. comp)


# perceptual network for calculation of style loss and perceptual loss
class PerceptualNetwork(nn.Module):
    def __init__(self):
        super(PerceptualNetwork, self).__init__()
        vgg_pretrained_features = vgg(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()

        for x in range(5):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(5, 10):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 17):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for i in range(1, 4):
            for param in getattr(self, 'slice{}'.format(i)).parameters():
                param.requires_grad = False

    def forward(self, input):
        pool1 = self.slice1(input)
        pool2 = self.slice2(pool1)
        pool3 = self.slice3(pool2)
        return [pool1, pool2, pool3]


def gram_matrix(y):
    n, c, h, w = y.size()
    features = y.view(n, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram


class BinaryFocalWeightedLoss(nn.Module):
    '''
    Binary Focal Weighted Loss

    '''

    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(BinaryFocalWeightedLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        # target serves as an index_selector
        # gather log_probability data from input by the guide of target
        logpt = F.log_softmax(input, 1)
        # select the log_probablity on input based on its index
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            # change alpha tensor's type same as input(torch.cudaFloatTensor:default)
            if self.alpha.type() != input.type():
                self.alpha = self.alpha.type_as(input)
            # at is a weight tensor for each class, its a 1-dim tensor (C,)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        # self.focalloss=BinaryFocalWeightedLoss()
        # self.perceptualnet = PerceptualNetwork()

    def forward(self, data_dict):
        ''''
        :param data_dict contains :
                data_dict['in']
                data_dict['out']
                data_dict['gt']
                data_dict['mask']
                data_dict['att']

        :return loss: a dict contains loss scalar
                '''
        loss_dict = {}
        if 'att' in data_dict.keys() and 'mask' in data_dict.keys():
            loss_dict['att'] = F.cross_entropy(data_dict['att'], data_dict['mask'],
                                               weight=torch.tensor([0.05, 0.95]).type_as(data_dict['att']))

        if 'in' in data_dict.keys() and 'out' in data_dict.keys() and 'gt' in data_dict.keys():
            comp = data_dict['mask'] * data_dict['in'] + (1.0 - data_dict['mask']) * data_dict['out']
            loss_dict['valid'] = F.l1_loss(data_dict['mask'] * data_dict['out'], data_dict['mask'] * data_dict['gt'])
            loss_dict['hole'] = 6.0 * F.l1_loss((1 - data_dict['mask']) * data_dict['out'],
                                                (1 - data_dict['mask']) * data_dict['gt'])
            feat_output = self.perceptualnet(data_dict['out'])
            feat_gt = self.perceptualnet(data_dict['gt'])
            feat_comp = self.perceptualnet(comp)
            loss_dict['content'] = 0.0
            for i in range(3):
                loss_dict['content'] += 0.05 * F.l1_loss(feat_output[i], feat_gt[i])
                loss_dict['content'] += 0.05 * F.l1_loss(feat_comp[i], feat_gt[i])
            loss_dict['style'] = 0.0
            for i in range(3):
                loss_dict['style'] += 120 * F.l1_loss(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i]))
                loss_dict['style'] += 120 * F.l1_loss(gram_matrix(feat_comp[i]), gram_matrix(feat_gt[i]))
            loss_dict['tv'] = 0.1 * torch.mean(torch.abs(comp[:, :, :, :-1] - comp[:, :, :, 1:])) + \
                              torch.mean(torch.abs(comp[:, :, :-1, :] - comp[:, :, 1:, :]))
        return loss_dict
