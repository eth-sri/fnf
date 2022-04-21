import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class FactorizedCategorical:

    def __init__(self, prior, dims):
        self.dims = dims
        self.prior = prior
        self.probs = []
        beg = 0
        for k in self.dims:
            self.probs += [Categorical(logits=prior[beg:beg+k])]
            beg += k

    def sample(self, n):
        ret = []
        for i, k in enumerate(self.dims):
            idx = self.probs[i].sample((n,))
            ret += [F.one_hot(idx, num_classes=k)]
        ret = torch.cat(ret, dim=1).float()
        return ret

    def log_prob(self, x):
        return (x * F.log_softmax(self.prior, dim=0)).sum(-1)
