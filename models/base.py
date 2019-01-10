from torch.nn import Module, init, Conv2d, BatchNorm2d
from collections import Iterable
from math import ceil


class Base(Module):
    def __init__(self):
        super(Base, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def initialization(self, modules):
        assert isinstance(modules, Iterable), 'modules should be wrapped in Iterable Collections.'
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, Conv2d) and layer.weight.requires_grad:
                    init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0.0)
                    # print('layer initialization finished on {}'.format(layer.__class__.__name__))
                if isinstance(layer, BatchNorm2d) and layer.weight.requires_grad:
                    init.constant_(layer.weight, 1.0)
                    init.constant_(layer.bias, 0.0)
                    # print('layer initialization finished on {}'.format(layer.__class__.__name__))

    def total_params(self):
        sum_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum_params, trainable_params

    @staticmethod
    def get_pad(input_size, kernel_size, stride, dilation=1):
        out_size = ceil(float(input_size) / stride)
        return int(((out_size - 1) * stride + dilation * (kernel_size - 1) + 1 - input_size) / 2)


if __name__ == '__main__':
    test = Base()
    print(test)
