import torch
import numpy as np
import torch.nn as nn

K = 100


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

    def __init__(self, D, hidden, mask=None):
        super(FlowLayer, self).__init__()
        self.D = D
        self.hidden = hidden
        self.s = Scale(D, hidden)
        self.t = Translate(D, hidden)
        mask = torch.from_numpy(mask)
        mask.requires_grad_(False)
        self.register_buffer('mask', mask)
        
    def forward(self, z):
        s = self.s(self.mask * z)
        t = self.t(self.mask * z)
        x = self.mask * z + (1 - self.mask) * (z * torch.exp(s) + t)
        logdet = torch.sum(s * (1 - self.mask), dim=1)
        return x, logdet

    def inverse(self, x):
        s = self.s(self.mask * x)
        t = self.t(self.mask * x)
        z = self.mask * x + (1 - self.mask) * ((x - t) * torch.exp(-s))
        logdet = torch.sum(s * (1 - self.mask), dim=1)
        return z, logdet


class BatchNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.batch_mean = None
        self.batch_var = None

    def inverse(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps
            self.set_batch_stats_func(x.detach())
        else:
            m = self.batch_mean
            v = self.batch_var

        z = (x - m) / torch.sqrt(v)
        z = z * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return z, log_det

    def forward(self, z):
        m = self.batch_mean
        v = self.batch_var
        x = (z - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x, log_det

    def set_batch_stats_func(self, x):
        if self.batch_mean is None:
            self.batch_mean = x.mean(dim=0)
            self.batch_var = x.var(dim=0) + self.eps
        else:
            self.batch_mean = 0.9 * self.batch_mean + 0.1 * x.mean(dim=0)
            self.batch_var = 0.9 * self.batch_var + 0.1 * (x.var(dim=0) + self.eps)


class FlowEncoder(nn.Module):

    def __init__(self, p_x, D, hidden, k, masks=None):
        super(FlowEncoder, self).__init__()
        self.p_x = p_x
        self.D = D
        self.hidden = hidden
        self.k = k
        if masks is None:
            masks = [
                np.array([(j + i) % 2 for j in range(D)])
                for i in range(k)
            ]
        self.layers = []
        for i in range(k):
            self.layers += [FlowLayer(D, hidden, masks[i])]
        self.layers = nn.ModuleList(self.layers)

    def inverse(self, x):
        if self.p_x is not None:
            sum_logp = self.p_x.log_prob(x)
        else:
            sum_logp = 0
        z = x
        for layer in reversed(self.layers):
            z, logdet = layer.inverse(z)
            sum_logp = sum_logp + logdet
        return z, sum_logp

    def forward(self, z):
        sum_logp = 0
        x = z
        for layer in self.layers:
            x, logdet = layer.forward(x)
            sum_logp = sum_logp + logdet
        if self.p_x is not None:
            sum_logp = sum_logp + self.p_x.log_prob(x)
        return x, sum_logp
