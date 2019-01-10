import torch.nn as nn
import torch
from loss import Criterion
from models.segmentation import VGG16_Inception_8s
from models.inpainting import Inpaint


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.segmentation = VGG16_Inception_8s(2)
        # self.inpaint=Inpaint()
        self.criterion = Criterion()

    def forward(self, in_dict):
        data_dict = {}
        if getattr(self, 'segmentation', None) is None and getattr(self, 'inpaint', None) is not None:
            output = self.inpaint((in_dict['origin'], in_dict['mask']))
            data_dict['in'] = in_dict['origin']
            data_dict['out'] = output
            data_dict['gt'] = in_dict['inpaint']
            data_dict['mask'] = in_dict['mask']
        # segmentation part
        if getattr(self, 'inpaint', None) is None and getattr(self, 'segmentation', None) is not None:
            attention = self.segmentation(in_dict['origin'])
            data_dict['in'] = in_dict['origin']
            data_dict['mask'] = in_dict['mask']
            data_dict['att'] = attention

        loss_dict = self.criterion(data_dict)
        return loss_dict, data_dict


def DataParallelModel(model, **kwargs):
    if 'device_ids' in kwargs.keys():
        device_ids = kwargs['device_ids']
    else:
        device_ids = None
    if 'output_device' in kwargs.keys():
        output_device = kwargs['output_device']
    else:
        output_device = None
    if 'cuda' in kwargs.keys():
        cudaID = kwargs['cuda']
        device = torch.device('cuda:{}'.format(cudaID))
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda()
    return model


if __name__ == '__main__':
    model = Model()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
