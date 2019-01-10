import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import os.path as path
from torchvision.utils import make_grid
import torch.nn.functional as F


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, models, optimizers, epoch):
    ckpt_dict = {'epoch': epoch}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)
    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load('./snapshot/ckpt/{:d}.pth'.format(ckpt_name))
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['epoch']


def check_data_name(data_path):
    names1 = [name.split('.')[0] for name in os.listdir(path.join(data_path, 'origin'))]
    names2 = [name.split('.')[0] for name in os.listdir(path.join(data_path, 'mask'))]
    names3 = [name.split('.')[0] for name in os.listdir(path.join(data_path, 'inpaint'))]
    names1.sort()
    names2.sort()
    names3.sort()
    for i in range(len(names1)):
        if names1[i] == names2[i] == names3[i]:
            pass
        else:
            print('no', i)
            return False
    return True


def calculate_meanstd(data_path):
    '''
    :returns mean std lists with order:[r,g,b]
    '''
    namelist = [name for name in os.listdir(os.path.join(data_path, 'origin'))]
    sum = 0.0
    sos = 0.0
    n = len(os.listdir(path.join(data_path, 'origin')))
    for filename in namelist:
        x = Image.open(path.join(data_path, 'origin', filename)).convert("RGB")
        x = np.array(x)
        x = x.astype(np.float)
        x = np.true_divide(x, 255.0)
        n += x.shape[0] * x.shape[1]
        sum_x = np.sum(x, (0, 1))
        sos_x = np.square(x)
        sos_x = np.sum(sos_x, (0, 1))
        sum += sum_x
        sos += sos_x
    mean = np.true_divide(sum, n)
    std = np.sqrt(np.subtract(np.true_divide(sos, n), np.square(mean)))
    return mean.tolist(), std.tolist()


def unnormalize(x, mean, std):
    mean = torch.tensor(mean, device=x.device)
    std = torch.tensor(std, device=x.device)
    x = x.transpose(1, 3)
    x = x * std + mean
    x = x.transpose(1, 3)
    return x


def visualization(out_dict):
    '''
    tensor.size() and tensor.shape returns same torch.Size object with different reference wrt its tensor
    :param kwargs: dict of image tensors to be visualized
    :return:
    '''
    result = []
    if 'in' in out_dict.keys():
        result.append(out_dict['in'])
    if 'att' in out_dict.keys():
        att = torch.argmax(F.softmax(out_dict['att'], 1), 1, keepdim=True)
        att = att.expand(out_dict['in'].size()).float()

        result.append(att)
    if 'mask' in out_dict.keys():
        mask = out_dict['mask'].clone()
        mask = mask.unsqueeze_(1).expand_as(out_dict['in']).float()
        result.append(mask)
    if 'out' in out_dict.keys():
        result.append(out_dict['out'])
    if 'gt' in out_dict.keys():
        result.append(out_dict['gt'])

    grid = make_grid(torch.cat(result))
    return grid
