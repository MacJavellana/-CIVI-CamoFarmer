import torch.nn as nn


class BasicConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=False,
                 bias=True,
                 up_size=0):

        super(BasicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels,
                                     eps=1e-5,
                                     momentum=0.01,
                                     affine=True)

        self.relu = None
        if relu:
            self.relu = nn.ReLU(inplace=True)

        self.up_sample = None
        self.up_size = up_size
        if up_size != 0:
            self.up_sample = nn.Upsample(size=(up_size, up_size),
                                         mode='bilinear')

    def forward(self, x):

        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        if self.up_size > 0:
            x = self.up_sample(x)

        return x


class Conv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 activation=True):

        super(Conv, self).__init__()

        layers = []

        layers += [nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=groups)]
        layers += [nn.BatchNorm2d(num_features=out_channels)]

        if activation:
            layers += [nn.LeakyReLU(negative_slope=0.1,
                                    inplace=True)]

        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class ReOrgLayer(nn.Module):

    def __init__(self,
                 stride):

        super(ReOrgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride

        x = x.view(batch_size,
                   channels,
                   _height,
                   self.stride,
                   _width,
                   self.stride).transpose(3, 4).contiguous()

        x = x.view(batch_size,
                   channels,
                   _height * _width,
                   self.stride * self.stride).transpose(2, 3).contiguous()

        x = x.view(batch_size,
                   channels,
                   self.stride * self.stride,
                   _height,
                   _width).transpose(1, 2).contiguous()

        x = x.view(batch_size,
                   -1,
                   _height,
                   _width)

        return x
