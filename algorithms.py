import torch
from nn import *
from main import UNSPECIFIED

class vanilla:
    def __init__(self, task, nn=UNSPECIFIED, lr=1e-4, **kwargs):
        if len(task.x_shape) == 1:
            self.nn = RegularNet(input_dim=task.x_shape[0], output_dim=0, **nn)
        else:
            self.nn = ConvNet(input_shape=task.x_shape, output_dim=0, **nn)
        self.opt = torch.optim.Adam(self.nn.parameters(), lr)

    def learn(self, step, x, y):
        guess = self.nn(x)
        loss = torch.nn.functional.mse_loss(guess, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return dict(loss=loss.detach().item())

    def predict(self, x):
        return self.nn(x)
