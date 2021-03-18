import math, torch, torchvision
import numpy, h5py
from main import UNSPECIFIED

class mnist:
    def __init__(self, dataset_size=UNSPECIFIED, seed=UNSPECIFIED, device=None):
        self.mnist_train = torchvision.datasets.MNIST("/tmp/mnist", download=True, train=True)
        self.mnist_test = torchvision.datasets.MNIST("/tmp/mnist", download=True, train=False)
        if dataset_size == UNSPECIFIED:
            self.dataset_size = self.mnist_train.data.shape[0]
        else:
            assert dataset_size <= self.dataset_size
            self.dataset_size = dataset_size
        self.random = numpy.random.RandomState(seed if seed != UNSPECIFIED else None)

    @property
    def x_shape(self):
        return [1, 28, 28]
    @property
    def x_min(self):
        return 0.
    @property
    def x_max(self):
        return 1.
    @property
    def target_n(self):
        return 10

    def data_preproc(self, x):
        return x[:,None,:,:].float()/255.
    def train_sample(self, n):
        idx = self.random.randint(0, self.dataset_size, [n])
        return self.data_preproc(self.mnist_train.data[idx]), self.mnist_train.targets[idx]
    def train_set_iterator(self, n):
        current_n = 0
        while current_n + n < self.dataset_size:
            idx = torch.arange(current_n, current_n+n)
            yield self.data_preproc(self.mnist_train.data[idx]), self.mnist_train.targets[idx], None
            current_n += n
    def test_set_iterator(self, n):
        current_n = 0
        while current_n + n < self.mnist_test.data.shape[0]:
            idx = torch.arange(current_n, current_n+n)
            yield self.data_preproc(self.mnist_test.data[idx]), self.mnist_test.targets[idx], None
            current_n += n

    def eval(self, step, guess, y, meta):
        error = (guess.argmax(-1) != y).double()
        return {"err": error.cpu().detach().tolist()}

