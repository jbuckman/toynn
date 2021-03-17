import torch, math
from torchvision.models import resnet18
from torch import nn
from functools import partial
import torch.nn.functional as F

class SaveableNN(nn.Module):
    def save(self, filename):
        torch.save({"definition": self.definition,
                    "state_dict": self.nn.state_dict()}, filename)
    def load(self, filename):
        spec = torch.load(filename)


class RegularNet(nn.Module):
    def __init__(self, input_dim, output_dim, layers=(32, 32, 64, 64), nonlin=nn.ReLU(inplace=True)):
        self.definition = ["RegularNet", (input_dim, output_dim, layers, nonlin)]
        if output_dim == 0: output_dim = 1; self.squeeze_out=True
        else: self.squeeze_out=False
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nonlin = nonlin
        self.layers = torch.nn.ModuleList([
            nn.Linear(in_d, out_d)
            for in_d, out_d in zip((input_dim,)+tuple(layers),tuple(layers)+(output_dim,))])

    def forward(self, x):
        x = self.nonlin(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = self.nonlin(layer(x))
        x = self.layers[-1](x)
        if self.squeeze_out: x = x[...,0]
        return x

    def all_forward(self, x):
        xs = [x]
        xs.append(self.layers[0](xs[-1]))
        xs.append(self.nonlin(xs[-1]))
        for layer in self.layers[1:-1]:
            xs.append(layer(xs[-1]))
            xs.append(self.nonlin(xs[-1]))
        xs.append(self.layers[-1](xs[-1]))
        if self.squeeze_out: xs[-1] = xs[-1][...,0]
        return [[x[i] for x in xs] for i in range(xs[0].shape[0])]

def conv2d_out_shape(in_shape, conv2d):
    C = conv2d.out_channels
    H = math.floor((in_shape[-2] + 2.*conv2d.padding[0] - conv2d.dilation[0] * (conv2d.kernel_size[0] - 1.) - 1.)/conv2d.stride[0] + 1.)
    W = math.floor((in_shape[-1] + 2.*conv2d.padding[1] - conv2d.dilation[1] * (conv2d.kernel_size[1] - 1.) - 1.)/conv2d.stride[1] + 1.)
    return [C, H, W]
cs = conv2d_out_shape

class ConvNet(nn.Module):
    def __init__(self, input_shape, output_dim, layers=(32, 32, 64, 64)):
        self.definition = ["ConvNet", (input_shape, output_dim, layers)]
        if output_dim == 0: output_dim = 1; self.squeeze_out=True
        else: self.squeeze_out=False
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=layers[0],
            kernel_size=8,
            stride=3,
            padding=4
        )
        self.conv2 = nn.Conv2d(layers[0], layers[1], kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(layers[1], layers[2], kernel_size=2, stride=1)
        self.shape_after_convs = cs(cs(cs(input_shape, self.conv1), self.conv2), self.conv3)
        self.fc = RegularNet(self.shape_after_convs[0]*self.shape_after_convs[1]*self.shape_after_convs[2], output_dim, layers=layers[3:])

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = x.view((x.shape[0], -1,))
        out = self.fc(x)
        if self.squeeze_out: out = out[...,0]
        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.GroupNorm(1, in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(1, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_channels=64):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        self.conv1 = nn.Conv2d(1, c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*c*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_shape, output_dim, size=5):
        super().__init__()
        self.definition = ["ResNet", (input_shape, output_dim, size)]
        self.nn = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=output_dim, init_channels=size)

    def forward(self, x):
        return self.nn(x)