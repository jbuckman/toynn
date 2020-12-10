import math, torch, torchvision

class sin_1d:
    def __init__(self, rate=20., amplitude=.5, dataset_size=None, train_range=[-1,1], test_range=[-2,2], **kwargs):
        if dataset_size is None: dataset_size = 20
        self.r = rate
        self.a = amplitude
        self.train_range = train_range
        self.test_range = test_range
        self.dataset = torch.rand([dataset_size,1])*(self.train_range[1] - self.train_range[0]) + self.train_range[0]
    @property
    def x_shape(self):
        return [1]
    @property
    def x_min(self):
        return self.test_range[0]
    @property
    def x_max(self):
        return self.test_range[1]
    @property
    def target_min(self):
        return -self.a
    @property
    def target_max(self):
        return self.a
    def f(self, x):
        return (self.a * torch.sin(self.r*math.pi*x))[...,0]
    def train_sample(self, n):
        x = self.dataset[torch.randint(len(self.dataset), (n,))]
        return x, self.f(x)
    def test_sample(self, n):
        x = torch.rand([n,1])*(self.test_range[1] - self.test_range[0]) + self.test_range[0]
        yield x, self.f(x)

class mnist:
    def __init__(self, dataset_size=None):
        self.mnist_train = torchvision.datasets.MNIST("/tmp/mnist", download=True, train=True)
        self.mnist_test = torchvision.datasets.MNIST("/tmp/mnist", download=True, train=False)
        self.dataset_size = self.mnist_train.data.shape[0]
        if dataset_size is not None:
            assert dataset_size <= self.dataset_size
            self.dataset_size = dataset_size
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
    def target_min(self):
        return 0.
    @property
    def target_max(self):
        return 9.
    def data_preproc(self, x):
        return x[:,None,:,:].float()/255.
    def target_preproc(self, y):
        return y.float()
    def train_sample(self, n):
        idx = torch.randint(0, self.dataset_size, [n])
        return self.data_preproc(self.mnist_train.data[idx]), self.target_preproc(self.mnist_train.targets[idx])
    def test_sample(self, n):
        current_n = 0
        while current_n + n < self.mnist_test.data.shape[0]:
            idx = torch.arange(current_n, current_n+n)
            yield self.data_preproc(self.mnist_test.data[idx]), self.target_preproc(self.mnist_test.targets[idx].float())
            current_n += n

class flat_mnist(mnist):
    @property
    def x_shape(self):
        return [1*28*28]
    def data_preproc(self, x):
        return super().data_preproc(x).reshape([-1] + self.x_shape)

