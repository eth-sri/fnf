import argparse
from generative.gmm import train_gmm
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from train_fnf import train_flow
from torch.distributions import MultivariateNormal

device = 'cuda'


def get_gaussians(y1, y2):
    mean1, cov1 = torch.FloatTensor([0, y1]), torch.FloatTensor([[1, 0], [0, 1]])
    mean2, cov2 = torch.FloatTensor([0, y2]), torch.FloatTensor([[1, 0], [0, 1]])
    p1 = MultivariateNormal(mean1.to(device), cov1.to(device))
    p2 = MultivariateNormal(mean2.to(device), cov2.to(device))
    target_fn = lambda x: (x[:, 0] > 0).long()
    return 2, p1, p2, target_fn, target_fn


def get_stripes(args, k, do_train_gmm=True):

    class GaussianMixture:

        def __init__(self, k, xs, ys, sigma):
            assert len(xs) == k
            self.p = []
            self.k = k
            for i in range(k):
                mean = torch.FloatTensor([xs[i], ys[i]])
                cov = sigma**2 * torch.FloatTensor([[1, 0], [0, 1]])
                self.p += [MultivariateNormal(mean.to(device), cov.to(device))]

        def sample(self, shape):
            n = shape[0]
            pre_samples = torch.cat([p.sample(shape).unsqueeze(1) for p in self.p], dim=1)
            select = torch.randint(0, self.k, (n, 1)).to(device)
            g = torch.zeros((n, self.k)).to(device)
            g.scatter_(1, select, 1)
            samples = (g.unsqueeze(2) * pre_samples).sum(1)
            return samples

        def log_prob(self, r):
            probs = []
            for i in range(self.k):
                probs += [self.p[i].log_prob(r).unsqueeze(0) + np.log(1.0 / self.k)]
            log_probs = torch.logsumexp(torch.cat(probs, dim=0), dim=0)
            return log_probs

    p1 = GaussianMixture(2, [-3, 3], [3, 3], 1.0)
    p2 = GaussianMixture(2, [-3, 3], [-3, -3], 1.0)

    def target_fn1(r):
        return (r[:, 0] > 0.0).long()

    def target_fn2(r):
        return (r[:, 0] < 0.0).long()

    train1, train2 = p1.sample((args.n_train, )), p2.sample((args.n_train, ))
    train1_targets, train2_targets = target_fn1(train1), target_fn2(train2)

    valid1, valid2 = p1.sample((args.n_valid, )), p2.sample((args.n_valid, ))
    valid1_targets, valid2_targets = target_fn1(valid1), target_fn2(valid2)

    train1_loader = DataLoader(TensorDataset(train1, train1_targets), batch_size=args.batch_size, shuffle=True)
    train2_loader = DataLoader(TensorDataset(train2, train2_targets), batch_size=args.batch_size, shuffle=True)

    valid1_loader = DataLoader(TensorDataset(valid1, valid1_targets), batch_size=args.batch_size, shuffle=True)
    valid2_loader = DataLoader(TensorDataset(valid2, valid2_targets), batch_size=args.batch_size, shuffle=True)

    if do_train_gmm:
        train = (train1, train2, train1_targets, train2_targets)
        valid = (valid1, valid2, valid1_targets, valid2_targets)
        prior1, prior2 = train_gmm(args, None, train, valid, device)
    else:
        prior1, prior2 = [None], [None]

    return 2, (p1, p2), (prior1[0], prior2[0]), (train1_loader, train2_loader), (valid1_loader, valid2_loader)


def compute_stat_dist(p1, p2, k, flows):
    data_x1 = p1.sample((k,)).to(device)
    data_x2 = p2.sample((k,)).to(device)

    # first
    x1_z1, x1_logp1 = flows[0][0].inverse(data_x1)
    x1_x2, x1_logp2 = flows[1][0].forward(x1_z1)

    x1_logp1 = x1_logp1 + p1.log_prob(data_x1)
    x1_logp2 = x1_logp2 + p2.log_prob(x1_x2)

    e1 = (x1_logp1 > x1_logp2).float().mean()

    # second
    x2_z2, x2_logp2 = flows[1][0].inverse(data_x2)
    x2_x1, x2_logp1 = flows[0][0].forward(x2_z2)

    x2_logp2 = x2_logp2 + p2.log_prob(data_x2)
    x2_logp1 = x2_logp1 + p1.log_prob(x2_x1)

    e2 = (x2_logp1 > x2_logp2).float().mean()
    return abs(e1-e2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--kl_start', type=int, default=0)
    parser.add_argument('--kl_end', type=int, default=200)
    parser.add_argument('--dec_epochs', type=int, default=100, help='num of epochs for decision')
    parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--adv_epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--n_blocks', type=int, default=10, help='num of blocks')
    parser.add_argument('--n_flows', type=int, default=1, help='num of flows')
    parser.add_argument('--plot', action='store_true', help='whether to plot')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_valid', type=int, default=1024)
    parser.add_argument('--log_epochs', type=int, default=10)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--fair_criterion', type=str, default='stat_parity', choices=['stat_parity', 'eq_odds', 'eq_opp'])
    parser.add_argument('--gmm_comps1', type=int, default=2)
    parser.add_argument('--gmm_comps2', type=int, default=2)
    parser.add_argument('--load_enc', action='store_true')
    parser.add_argument('--scalarization', type=str, default='convex', choices=['convex', 'chebyshev'])
    parser.add_argument('--save_enc', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    global device
    device = args.device

    # d, p1, p2, target_fn = get_gaussians(3.5, -3.5)
    d, (p1_true, p2_true), (p1_hat, p2_hat), train_loaders, valid_loaders = get_stripes(args, 4)
    clf_dims = [100]
    flow_dims = [50, 50]

    # p1_true = GMM(p1_true, device)
    # p2_true = GMM(p2_true, device)

    tv_samples = 50000

    tv1 = p1_hat.total_variation(p1_true, n_samples=tv_samples)
    tv2 = p2_hat.total_variation(p2_true, n_samples=tv_samples)

    print('tv: ', tv1, tv2)

    # flows = train_flow(args, d, None, [p1_true], [p2_true], flow_dims, clf_dims, train_loaders, valid_loaders, None)
    flows = train_flow(args, d, None, [p1_hat], [p2_hat], flow_dims, clf_dims, train_loaders, valid_loaders, None)

    k = 50000
    stat_dist_true = compute_stat_dist(p1_true, p2_true, k, flows)
    stat_dist_hat = compute_stat_dist(p1_hat, p2_hat, k, flows)
    print('stat_dist_true: ', stat_dist_true)
    print('stat_dist_hat: ', stat_dist_hat)
    exit(0)
    
    data_x1 = p1_true.sample((k,)).to(device)
    data_x2 = p2_true.sample((k,)).to(device)
    x1_z1, log_prob1 = flows[0][0].inverse(data_x1)
    x2_z2, log_prob2 = flows[1][0].inverse(data_x2)
    e1 = flows[0][0].log_prob(x1_z1)
    exit(0)

    
    k = 1000
    
    data_x1 = p1_true.sample((k,)).to(device)
    data_x2 = p2_true.sample((k,)).to(device)

    x1_z1, _ = flows[0][0].inverse(data_x1)
    x2_z2, _ = flows[1][0].inverse(data_x2)

    x1_z1 = x1_z1.detach().cpu().numpy()
    x2_z2 = x2_z2.detach().cpu().numpy()

    # plt.scatter(x1_z1[:, 0], x1_z1[:, 1], s=5)
    # plt.scatter(x2_z2[:, 0], x2_z2[:, 1], s=5)
    # plt.show()


if __name__ == '__main__':
    main()
