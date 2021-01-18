import torch.nn as nn
import torch


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, channel, stride, down_sample=False):
        super(Bottleneck, self).__init__()
        self.down_sample = down_sample
        self.in_channel = in_channel
        self.channel = channel
        # 因为同一个layer中的block有“是否执行下采样”的区别
        # 所以需要把stride作为block的一个参数传入
        self.stride = stride

        # kernel_size=1, stride=1, padding=0
        # 这样的参数组合可以使特征图大小不改变
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=channel,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU()

        # kernel_size=3, stride=1, padding=1

        # 这样的参数组合可以使特征图大小不改变
        # 传入的stride用来控制第一个conv的stride，因为block中的下采样操作由第一个conv3x3来执行
        self.conv2 = nn.Conv2d(in_channels=channel,
                               out_channels=channel,
                               kernel_size=3,
                               stride=self.stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=channel,
                               out_channels=channel * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.bn3 = nn.BatchNorm2d(channel * self.expansion)
        self.relu3 = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.down_sample:
            # residual = self.max_pool(residual)

            # kernel_size=3, stride=2, padding=1
            # 这样的参数组合可以实现2倍的下采样
            # downsample的输入输出通道数与整个block的输入输出通道数保持一致
            # downsample的stride与block中负责降采样的那个conv的stride保持一致
            down_sample = nn.Sequential(nn.Conv2d(in_channels=self.in_channel,
                                                  out_channels=self.channel * self.expansion,
                                                  kernel_size=3,
                                                  stride=self.stride,
                                                  padding=1),
                                        nn.BatchNorm2d(self.channel * self.expansion))
            residual = down_sample(residual)
        print("residual.shape", residual.shape)
        print("x.shape", x.shape)
        out = residual + x
        out = self.relu3(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, channel, stride, down_sample=False):
        super(BasicBlock, self).__init__()
        self.down_sample = down_sample
        self.in_channel = in_channel
        self.channel = channel
        # 因为同一个layer中的block有“是否执行下采样”的区别
        # 所以需要把stride作为block的一个参数传入
        self.stride = stride

        # 传入的stride用来控制第一个conv的stride，因为block中的下采样操作由第一个conv来执行
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=channel,
                               kernel_size=3,
                               stride=self.stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        # 为什么官方代码中的relu可以复用，都是直接一个self.relu = nn.ReLU()用全场？
        # 是因为relu里边没有可学习的参数吗？
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=channel,
                               out_channels=channel,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(channel)

        self.relu2 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.down_sample:  # 官方代码是用卷积来做下采样
            # residual = self.max_pool(residual)

            # downsample的输入输出通道数与整个block的输入输出通道数保持一致
            # downsample的stride与block中负责降采样的那个conv的stride保持一致
            down_sample = nn.Sequential(nn.Conv2d(in_channels=self.in_channel,
                                                  out_channels=self.channel,
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1),
                                        nn.BatchNorm2d(self.channel))
            residual = down_sample(residual)

        print("residual.shape: ", residual.shape)
        print("x.shape: ", x.shape)

        out = residual + x
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        layer_1 = []
        for i in range(layers[0]):
            layer_1.append(block(self.in_channel, 64, 1, False))

        self.layer_1 = self._make_layer(block, layers[0], 64, stride=1)
        print(self.in_channel)
        self.layer_2 = self._make_layer(block, layers[1], 128, stride=2)
        print(self.in_channel)
        self.layer_3 = self._make_layer(block, layers[2], 256, stride=2)
        print(self.in_channel)
        self.layer_4 = self._make_layer(block, layers[3], 512, stride=2)
        print(self.in_channel)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=1000)

    def _make_layer(self, block, block_num, channel, stride=1):
        layer = []
        # for BasicBlock
        if self.in_channel == channel * block.expansion:
            if stride == 1:  # layer_1
                # 因为整个block的输入输出通道数和特征图尺寸保持一致
                # 所不需要执行downsample，所以是False
                layer.append(block(self.in_channel, channel, 1, False))
            else:  # layer_2 to layer_4
                layer.append(block(self.in_channel, channel, 2, True))

        # for Bottleneck
        elif self.in_channel != channel * block.expansion:
            if stride == 1:  # layer_1
                # 只升维，不做降采样
                layer.append(block(self.in_channel, channel, 1, True))
            else:  # layer_2 to layer_4
                # 使用conv1x1降维，使用conv3x3降采样，使用residual升维
                layer.append(block(self.in_channel, channel, 2, True))

        self.in_channel = channel * block.expansion

        for i in range(1, block_num):
            layer.append(block(self.in_channel, channel, 1, False))
        layer = nn.Sequential(*layer)

        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        print("========================= layer_1 =========================")
        x = self.layer_1(x)
        print("========================= layer_2 =========================")
        x = self.layer_2(x)
        print("========================= layer_3 =========================")
        x = self.layer_3(x)
        print("========================= layer_4 =========================")
        x = self.layer_4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        out = self.fc(x)

        return out


def resnet18():
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2])


def resnet34():
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3])


def resnet50():
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3])


def resnet101():
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3])


def resnet152():
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3])


def _resnet(arch, block, layers):
    model = ResNet(block, layers)
    return model


if __name__ == '__main__':
    num_classes = 4
    x = torch.rand([4, 3, 224, 224])
    model = resnet152()
    # print(model)
    out = model(x)
