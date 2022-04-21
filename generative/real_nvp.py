import torch
import numpy as np
import torch.nn as nn

K = 200


def get_layers(D, hidden):
    hidden = hidden + [D]
    ret = [nn.Linear(D, hidden[0])]
    for i in range(len(hidden) - 1):
        ret += [nn.ReLU(), nn.Linear(hidden[i], hidden[i+1])]
    return ret


class Scale(nn.Module):

    def __init__(self, D, hidden):
        super(Scale, self).__init__()
        layers = get_layers(D, hidden) + [nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Translate(nn.Module):

    def __init__(self, D, hidden):
        super(Translate, self).__init__()
        layers = get_layers(D, hidden)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlowLayer(nn.Module):

    def __init__(self, D, hidden, mask):
        super(FlowLayer, self).__init__()
        self.D = D
        self.s = Scale(D, hidden)
        self.t = Translate(D, hidden)
        mask = torch.from_numpy(mask)
        mask.requires_grad_(False)
        self.register_buffer('mask', mask)
        
    def forward(self, z):
        s = self.s(self.mask * z)
        t = self.t(self.mask * z)
        x = self.mask * z + (1 - self.mask) * (z * torch.exp(s) + t)
        return x

    def inverse(self, x):
        s = self.s(self.mask * x)
        t = self.t(self.mask * x)
        z = self.mask * x + (1 - self.mask) * ((x - t) * torch.exp(-s))
        logdet = torch.sum(-s * (1 - self.mask), dim=1)
        return z, logdet


class LogLayer(nn.Module):

    def __init__(self):
        super(LogLayer, self).__init__()

    def forward(self, z):
        x = z.log()
        return x

    def inverse(self, x):
        z = x.exp()
        logdet = torch.sum(x, dim=1)
        return z, logdet


class MultiGumbel:

    def __init__(self, g):
        self.g = g

    def log_prob(self, z):
        return self.g.log_prob(z).sum(1)

    def sample(self, shape):
        return self.g.sample(shape)


class FlowNetwork(nn.Module):

    def __init__(self, p_z, D, hidden, k, masks=None):
        super(FlowNetwork, self).__init__()
        self.p_z = p_z
        self.D = D
        self.hidden = hidden
        self.k = k

        if masks is None:
            masks = [
                np.array([[(i + j) % 2 for j in range(D)]])
                for i in range(k)
            ]

        self.layers = []
        for i in range(k):
            self.layers += [FlowLayer(D, hidden, masks[i])]
        # self.layers = [LogLayer()] + self.layers
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, x):
        sum_logp = 0
        z = x
        for layer in reversed(self.layers):
            z, logdet = layer.inverse(z)
            sum_logp = sum_logp + logdet
        sum_logp = sum_logp + self.p_z.log_prob(z)
        return z, sum_logp

    def log_prob(self, x):
        return self.inverse(x)[1]

    def sample(self, shape):
        z = self.p_z.sample(shape)
        return self.forward(z)
