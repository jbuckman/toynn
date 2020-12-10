import torch
from nn import *

class vanilla:
    def __init__(self, task, **kwargs):
        nn_kwargs = {k[3:]: v for k, v in kwargs.items() if k[:3] == "nn_"}
        if len(task.x_shape) == 1:
            self.nn = RegularNet(input_dim=task.x_shape[0], output_dim=0, **nn_kwargs)
        else:
            self.nn = ConvNet(input_shape=task.x_shape, output_dim=0, **nn_kwargs)
        self.opt = torch.optim.Adam(self.nn.parameters(), 1e-4)

    def learn(self, step, x, y):
        guess = self.nn(x)
        loss = torch.nn.functional.mse_loss(guess, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return dict(loss=loss.detach().item())

    def predict(self, x):
        return self.nn(x)
