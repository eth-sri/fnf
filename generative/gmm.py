import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
from sklearn import mixture
from sklearn.neighbors import KernelDensity


# Wrapper for GMM so that it has same interface as RealNVP
class GMM(nn.Module):

    def __init__(self, gmm, device):
        super(GMM, self).__init__()
        self.weights = torch.from_numpy(gmm.weights_).float().to(device)
        self.means = torch.from_numpy(gmm.means_).float().to(device)
        self.covs = torch.from_numpy(gmm.covariances_).float().to(device)

        if len(self.covs.shape) == 2:
            self.covs = [torch.diag(self.covs[i]).unsqueeze(0) for i in range(self.covs.shape[0])]
            self.covs = torch.cat(self.covs, dim=0)

        g = D.MultivariateNormal(self.means, self.covs)
        mix = D.Categorical(self.weights)
        self.gmm = D.MixtureSameFamily(mix, g)

    def sample(self, shape):
        return self.gmm.sample(shape)

    def log_p(self, x):
        return self.gmm.log_prob(x)

    def log_prob(self, x):
        return self.gmm.log_prob(x)

    def total_variation(self, gmm2, n_samples):
        x1 = self.sample((n_samples, ))
        x2 = gmm2.sample((n_samples, ))
        t1 = 2*(self.log_prob(x1) > gmm2.log_prob(x1)).float() - 1
        t2 = 2*(self.log_prob(x2) > gmm2.log_prob(x2)).float() - 1
        return abs(t1.mean() - t2.mean())


def train_gmm(args, q, train, valid, device):
    train1, train2, targets1, targets2 = train
    valid1, valid2, v_targets1, v_targets2 = valid
    print('gmm_comps: ', args.gmm_comps1, args.gmm_comps2)
    if args.fair_criterion == 'stat_parity' or args.fair_criterion == 'eq_opp':
        prior1 = [mixture.GaussianMixture(n_components=args.gmm_comps1, n_init=1, covariance_type='full')]
        prior2 = [mixture.GaussianMixture(n_components=args.gmm_comps2, n_init=1, covariance_type='full')]
        if q is not None:
            t1 = torch.clamp(train1 + q * torch.rand(train1.shape).to(device), args.alpha/2, 1-args.alpha/2).logit().detach().cpu().numpy()
            t2 = torch.clamp(train2 + q * torch.rand(train2.shape).to(device), args.alpha/2, 1-args.alpha/2).logit().detach().cpu().numpy()
            v1 = torch.clamp(valid1 + q * torch.rand(valid1.shape).to(device), args.alpha/2, 1-args.alpha/2).logit().detach().cpu().numpy()
            v2 = torch.clamp(valid2 + q * torch.rand(valid2.shape).to(device), args.alpha/2, 1-args.alpha/2).logit().detach().cpu().numpy()
        else:
            t1, t2 = train1.detach().cpu().numpy(), train2.detach().cpu().numpy()
            v1, v2 = valid1.detach().cpu().numpy(), valid2.detach().cpu().numpy()
        if args.fair_criterion == 'eq_opp':
            t1 = t1[targets1.detach().cpu().numpy() == 1]
            t2 = t2[targets2.detach().cpu().numpy() == 1]
        prior1[0].fit(t1)
        prior2[0].fit(t2)

        print(prior1[0].score(t1), prior2[0].score(t2))
        print(prior2[0].score(t1), prior1[0].score(t2))
        print('====')
        print(prior1[0].score(v1), prior2[0].score(v2))
        print(prior2[0].score(v1), prior1[0].score(v2))
    else:
        prior1 = [mixture.GaussianMixture(n_components=args.gmm_comps1, covariance_type='full'),
                  mixture.GaussianMixture(n_components=args.gmm_comps1, covariance_type='full')]
        prior2 = [mixture.GaussianMixture(n_components=args.gmm_comps2, covariance_type='full'),
                  mixture.GaussianMixture(n_components=args.gmm_comps2, covariance_type='full')]
        t1 = torch.clamp(train1 + q * torch.rand(train1.shape).to(device), args.alpha/2, 1-args.alpha/2).logit().detach().cpu().numpy()
        t2 = torch.clamp(train2 + q * torch.rand(train2.shape).to(device), args.alpha/2, 1-args.alpha/2).logit().detach().cpu().numpy()
        for y in range(2):
            y_t1 = t1[targets1.detach().cpu().numpy() == y]
            y_t2 = t2[targets2.detach().cpu().numpy() == y]
            prior1[y].fit(y_t1)
            prior2[y].fit(y_t2)
    for i in range(len(prior1)):
        prior1[i], prior2[i] = GMM(prior1[i], device), GMM(prior2[i], device)

    return prior1, prior2
