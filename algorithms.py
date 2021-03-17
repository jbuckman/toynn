import torch
from nn import *
from main import UNSPECIFIED

class vanilla:
    def __init__(self, task, device=None, nn=UNSPECIFIED, resnet=True, lr=1e-4, **kwargs):
        self.task = task
        if resnet:
            self.nn = ResNet(input_shape=task.x_shape, output_dim=task.target_n, **nn)
        else:
            self.nn = ConvNet(input_shape=task.x_shape, output_dim=task.target_n, **nn)
        self.nn.to(device)
        self.opt = torch.optim.Adam(self.nn.parameters(), lr, amsgrad=True)

    def learn(self, step, x, y):
        guess = self.nn(x)
        loss = torch.nn.functional.cross_entropy(guess, y, reduction='mean')
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return dict(loss=loss.cpu().detach().item())

    def eval(self, step, x, y, meta):
        guess = self.nn(x)
        loss = torch.nn.functional.cross_entropy(guess, y, reduction='none')
        task_eval = self.task.eval(step, guess, y, meta)
        return dict(loss=loss.cpu().detach().tolist(), **task_eval)

    def predict(self, x):
        return self.nn(x)
