import torch
from torch import nn
from model.registry import register_model


class VGG(nn.Module):
    def __init__(self, layers):
        super(VGG, self).__init__()

        self.inplanes = 3

        self.layer_1 = self._make_layer(layers[0], 64)
        self.layer_2 = self._make_layer(layers[1], 128)
        self.layer_3 = self._make_layer(layers[2], 256)
        self.layer_4 = self._make_layer(layers[3], 512)
        self.layer_5 = self._make_layer(layers[4], 512)

        self.fcs = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                 nn.Linear(4096, 4096),
                                 nn.Linear(4096, 1000))

        # self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, num, out_channel):
        """
        kernel_size=3, stride=1, padding=1
        这样的参数组合可以使输入输出的尺寸一致，即
        H_out = (H_in + 2 * padding - kernel_size) / 1 + 1 = (H_in + 2 * 1 - 3) / 1 + 1 = H_in
        :param num:
        :param out_channel:
        :return:
        """
        layer = []
        for i in range(num):
            layer.append(nn.Conv2d(in_channels=self.inplanes,
                                   out_channels=out_channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1))
            layer.append(nn.ReLU())
            if i == 0:
                self.inplanes = out_channel

        layer.append(nn.MaxPool2d(kernel_size=2))
        layer = nn.Sequential(*layer)
        return layer

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        # x = torch.reshape(x, (x.size(0), -1,))
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        # x = self.softmax(x)

        return x

@register_model
def vgg11():
    return _vgg('vgg16', [1, 1, 2, 2, 2])

@register_model
def vgg13():
    return _vgg('vgg16', [2, 2, 2, 2, 2])

@register_model
def vgg16():
    return _vgg('vgg16', [2, 2, 3, 3, 3])

@register_model
def vgg19():
    return _vgg('vgg16', [2, 2, 4, 4, 4])


def _vgg(arch, layers):
    model = VGG(layers)
    return model


if __name__ == '__main__':
    num_classes = 4
    x = torch.rand([4, 3, 224, 224])

    model = vgg16()
    feature = model.fcs[2].in_features
    model.fcs[2] = nn.Linear(feature, num_classes, bias=True)

    out = model(x)
