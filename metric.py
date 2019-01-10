import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score


def precision(input, target):
    '''
    :param input: 1x1xhxw
    :param target: 1xhxw
    :return:precision score
    '''
    input = input.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    input = np.reshape(input, (-1))
    target = np.reshape(target, (-1))
    return precision_score(input, target)


def recall(input, target):
    '''

    :param input:
    :param target:
    :return:
    '''
    input = input.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    input = np.reshape(input, (-1))
    target = np.reshape(target, (-1))
    return recall_score(input, target)
