from torchvision.models.vgg import vgg16, vgg16_bn
from models.base import Base
import torch.nn.functional as F
from torch.nn import Sequential, Conv2d, ReLU, Sigmoid, Module, BatchNorm2d
import torch


class Inception(Base):
    def __init__(self, in_channels, out_channels, sample=None):
        super(Inception, self).__init__()
        inter_channels = round(in_channels // 4)
        if sample == 'down':
            self.sample = Sequential(
                Conv2d(out_channels, out_channels, 3, 2, 1),
                BatchNorm2d(out_channels),
                ReLU(True)
            )
        elif sample == 'up':
            self.sample = Sequential(
                Upsample(),
                Conv2d(out_channels, out_channels, 3, 1, 1),
                BatchNorm2d(out_channels),
                ReLU(True)
            )
        else:
            self.sample = None
        self.link0 = Sequential(
            Conv2d(in_channels, inter_channels, 1),
            BatchNorm2d(inter_channels),
            ReLU(True)
        )
        self.link1 = Sequential(
            Conv2d(in_channels, inter_channels, 1),
            Conv2d(inter_channels, inter_channels, 3, 1, 1),
            BatchNorm2d(inter_channels),
            ReLU(True)
        )
        self.link2 = Sequential(
            Conv2d(in_channels, inter_channels, 1),
            Conv2d(inter_channels, inter_channels, 5, 1, 2),
            BatchNorm2d(inter_channels),
            ReLU(True)
        )
        self.link3 = Sequential(
            Conv2d(in_channels, inter_channels, 1),
            Conv2d(inter_channels, inter_channels, 3, 1, 2, dilation=2),
            BatchNorm2d(inter_channels),
            ReLU(True)
        )
        self.conv_linear = Conv2d(4 * inter_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.link0(x)
        x1 = self.link1(x)
        x2 = self.link2(x)
        x3 = self.link3(x)
        out = self.conv_linear(torch.cat([x0, x1, x2, x3], 1))
        if self.sample is not None:
            out = self.sample(out)
        return out


class Upsample(Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2)


class VGG16_Inception_8s(Base):
    def __init__(self, num_classes):
        super(VGG16_Inception_8s, self).__init__()
        self.num_classes = num_classes
        self.backbone = self._make_feature_encoder()
        self.decoder = self._make_feature_decoder()
        self.in_3 = Inception(256, 64, 'down')
        self.in_4 = Inception(512, 128)
        self.in_5 = Inception(512, 64, 'up')
        self.fc = self._make_fc_layers()
        self.initialization([self.in_3, self.in_4, self.in_5, self.fc])

    def _make_feature_encoder(self):
        model = vgg16_bn(pretrained=True).features[:-1]
        layers = Sequential()
        layers.add_module('b1', model[:6])
        layers.add_module('b2', model[6:13])
        layers.add_module('b3', model[13:23])
        layers.add_module('b4', model[23:33])
        layers.add_module('b5', model[33:])
        return layers

    def _make_feature_decoder(self):
        layers = Sequential(
            Sequential(
                Upsample(),
                Conv2d(256, 256, 3, 1, 1),
                BatchNorm2d(256),
                ReLU(True)
            ),
            Sequential(
                Upsample(),
                Conv2d(256, 128, 3, 1, 1),
                BatchNorm2d(128),
                ReLU(True)
            ),
            Sequential(
                Upsample(),
                Conv2d(128, 64, 3, 1, 1),
                BatchNorm2d(64),
                ReLU(True)
            )
        )
        return layers

    def _make_fc_layers(self):
        layer = Sequential(
            Conv2d(64, self.num_classes, 1),
            Sigmoid()
        )
        return layer

    def forward(self, x):
        feat = [x]
        for index, block in enumerate(self.backbone):
            feat.append(block(feat[-1]))
        mim = torch.cat([self.in_3(feat[3]), self.in_4(feat[4]), self.in_5(feat[5])], 1)
        out = self.decoder(mim)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    test = VGG16_Inception_8s(2)
    x = torch.ones((1, 3, 512, 512))
    y = test(x)
    print(y.shape)
    print(test.total_params())
