import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

n_epochs = 10000
batch_size = 64
device = 'cpu'


class MaskedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask = torch.from_numpy(mask).float().to(device)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

        
class MADE(nn.Module):

    def __init__(self, in_dim, hidden_sz, m_first=None, shuffle_m=False):
        super(MADE, self).__init__()
        self.in_dim = in_dim
        self.hidden_sz = hidden_sz
        self.m = {}

        self.layers = []
        masked_linear = []
        hs = [in_dim] + self.hidden_sz + [in_dim]
        for i in range(1, len(hs)):
            if i > 1:
                self.layers += [nn.ReLU()]
            layer = MaskedLinear(hs[i-1], hs[i])
            self.layers += [layer]
            masked_linear += [layer]
        self.layers = nn.Sequential(*self.layers)

        all_ids = list(range(in_dim))
        if shuffle_m:
            np.random.shuffle(all_ids)
        if m_first is not None:
            all_ids.sort(key=lambda v: (v not in m_first))
        self.m[-1] = np.array([all_ids.index(i) for i in range(in_dim)])
        
        for i in range(len(self.hidden_sz)):
            prev_min = np.min(self.m[i-1])
            self.m[i] = np.random.randint(prev_min, self.in_dim-1, size=self.hidden_sz[i])

        masks = [
            self.m[i-1][:, None] <= self.m[i][None, :]
            for i in range(len(self.hidden_sz))
        ]
        masks += [self.m[len(self.hidden_sz)-1][:, None] < self.m[-1][None, :]]
            
        for i, layer in enumerate(masked_linear):
            layer.set_mask(masks[i].T)

        self.direct = MaskedLinear(in_dim, in_dim)
        self.direct.set_mask(self.m[-1][:, None] > self.m[-1][None, :])

    def forward(self, x):
        return self.direct(x) + self.layers(x)

    def log_prob(self, x):
        y = self.forward(x)
        return -F.binary_cross_entropy_with_logits(y, x, reduction='none').sum(-1)

    def sample(self, n_samples, device='cuda'):
        x = torch.zeros((n_samples, self.in_dim)).to(device)
        for j in range(self.in_dim):
            i = self.m[-1][j]
            y = torch.sigmoid(self.forward(x))[:, i]
            x[:, i] = (torch.rand(y.shape).to(device) < y)
        return x
