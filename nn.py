import torch, math
from torch import nn

class RegularNet(nn.Module):
    def __init__(self, input_dim, output_dim, layers=(32, 32, 64, 64), nonlin=nn.ReLU(inplace=True)):
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
