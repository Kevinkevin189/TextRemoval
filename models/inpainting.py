import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, LeakyReLU, init, BatchNorm2d
import torch.nn.functional as F


class PartialConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 batch_norm=True, activation='relu'):
        super(PartialConv, self).__init__()
        self.feat_conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        if batch_norm:
            self.batch_norm = BatchNorm2d(out_channels)
        else:
            self.batch_norm = None
        if activation == 'relu':
            self.activation = ReLU(True)
        elif activation == 'leaky':
            self.activation = LeakyReLU(0.2, True)
        else:
            self.activation = None
        init.kaiming_normal_(self.feat_conv.weight)
        init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, args):
        # feat,mask=args
        # feat=self.feat_conv(feat * mask)
        # with torch.no_grad():
        #     mask=self.mask_conv(mask)#mask sums
        # #mark holes in the mask, 1 for holes
        # holes=mask==0
        # inter_mask=mask.clone()
        # #fill inter mask 0 with 1 for division
        # inter_mask.masked_fill_(holes,1.0)
        # feat=(feat/inter_mask).masked_fill_(holes,0.0)
        # mask=(mask==1).float()
        # if self.batch_norm is not None:
        #     feat=self.batch_norm(feat)
        # if self.activation is not None:
        #     feat=self.activation(feat)
        # return (feat,mask)
        x, mask = args
        output = self.feat_conv(x * mask)
        if self.feat_conv.bias is not None:
            output_bias = self.feat_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)
        #
        with torch.no_grad():
            output_mask = self.mask_conv(mask)  # mask sums

        no_update_holes = output_mask == 0
        # because those values won't be used , assign a easy value to compute
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        return output, new_mask


class Upsample(Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, args):
        x, mask = args
        return F.interpolate(x, scale_factor=2), F.interpolate(mask, scale_factor=2)


class Inpaint(Module):
    def __init__(self):
        super(Inpaint, self).__init__()
        self.enc_cfg = [
            [3, 64, 7, 2, 3, 1, 1, True, False, 'relu'],
            [64, 128, 5, 2, 2, 1, 1, True, False, 'relu'],
            [128, 256, 5, 2, 2, 1, 1, True, False, 'relu'],
            [256, 512, 3, 2, 1, 1, 1, True, False, 'relu'],
            [512, 512, 3, 2, 1, 1, 1, True, False, 'relu'],
            [512, 512, 3, 2, 1, 1, 1, True, False, 'relu'],
            [512, 512, 3, 2, 1, 1, 1, True, False, 'relu'],
            [512, 512, 3, 2, 1, 1, 1, True, False, 'relu']
        ]

        self.dec_cfg = [
            [512 + 512, 512, 3, 1, 1, 1, 1, True, False, 'leaky'],
            [512 + 512, 512, 3, 1, 1, 1, 1, True, False, 'leaky'],
            [512 + 512, 512, 3, 1, 1, 1, 1, True, False, 'leaky'],
            [512 + 512, 512, 3, 1, 1, 1, 1, True, False, 'leaky'],
            [512 + 256, 256, 3, 1, 1, 1, 1, True, False, 'leaky'],
            [256 + 128, 128, 3, 1, 1, 1, 1, True, False, 'leaky'],
            [128 + 64, 64, 3, 1, 1, 1, 1, True, False, 'leaky'],
            [64 + 3, 3, 3, 1, 1, 1, 1, True, False, None]
        ]
        self.encoder = self.make_layers(self.enc_cfg)
        self.decoder = self.make_layers(self.dec_cfg)
        self.upsample = Upsample()

    def make_layers(self, settings):
        layer = []
        for in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation in settings:
            # print(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,batch_norm,activation)
            layer.append(
                PartialConv(in_channels, out_channels, kernel_size, stride, padding, dilation,
                            groups, bias,
                            batch_norm, activation))
        return Sequential(*layer)

    def forward(self, args):
        x, mask = args
        feature_x, feature_mask = [x], [mask]
        for index, layer in enumerate(self.encoder):
            x, mask = layer((x, mask))
            feature_x.append(x)
            feature_mask.append(mask)
        feature_x = feature_x[:-1]
        feature_mask = feature_mask[:-1]

        for layer in self.decoder:
            x_up, mask_up = self.upsample((x, mask))
            x_h = torch.cat([x_up, feature_x.pop(-1)], dim=1)
            mask_h = torch.cat([mask_up, feature_mask.pop(-1)], dim=1)
            x, mask = layer((x_h, mask_h))
        return x


if __name__ == '__main__':
    test = Inpaint()
    print(sum(p.numel() for p in test.parameters() if p.requires_grad))
    x = torch.ones((1, 3, 512, 512))
    m = torch.ones((1, 3, 512, 512))
    feat = test((x, m))
    print(feat.shape)

    '''
    debug status: 
        partial_conv: runs well only without sequential wrapped,but after I changed the input to args(input,mask),it solved.
        depth_partial_conv: runs well only if enmerate layer sequence
        inpaint network: won't run in any case
    '''
