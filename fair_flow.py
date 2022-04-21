import argparse
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from real_nvp_encoder import FlowEncoder
from torch.distributions import MultivariateNormal

from train_fnf import train_flow

device = 'cpu'

class ShiftFlow(nn.Module):

    def __init__(self, prior, d):
        super(ShiftFlow, self).__init__()
        self.prior = prior
        self.d = d
        self.shift = nn.Parameter(torch.randn(1, 2).to(device))
        # self.shift = nn.Parameter(torch.FloatTensor([[0, -4]]).to(device))

    def inverse(self, x):
        return x + self.shift, self.prior.log_prob(x)

    def forward(self, z):
        x = z - self.shift
        return x, self.prior.log_prob(x)

class FlipFlow(nn.Module):

    def __init__(self, prior, d):
        super(FlipFlow, self).__init__()
        self.prior = prior
        self.d = d
        self.mul = torch.FloatTensor([[0, -1]]).to(device)

    def inverse(self, x):
        z = x.clone()
        z[:, 0] = 9 - x[:, 0]
        return z, self.prior.log_prob(x)

    def forward(self, z):
        x = z.clone()
        x[:, 0] = 9 - z[:, 0]
        return x, self.prior.log_prob(x)


# def train_flow(args, d, p1, p2, target_fn1, target_fn2):
#     flow1 = FlowEncoder(p1, d, [20, 20], args.n_blocks).to(device)
#     flow2 = FlowEncoder(p2, d, [20, 20], args.n_blocks).to(device)
#     # flow2.load_state_dict(flow1.state_dict())
#     # flow1 = ShiftFlow(p1, d)
#     # flow2 = FlipFlow(p2, d)
#     batch_size = 512

#     clf = nn.Sequential(
#         nn.Linear(2, 50),
#         nn.ReLU(),
#         nn.Linear(50, 50),
#         nn.ReLU(),
#         nn.Linear(50, 2)
#     ).to(device)
    
#     opt = optim.Adam(list(clf.parameters()) + list(flow1.parameters()) + list(flow2.parameters()), lr=1e-2)

#     for epoch in range(args.n_epochs):
#         opt.zero_grad()
#         data_x1 = p1.sample((batch_size,)).to(device)
#         data_x2 = p2.sample((batch_size,)).to(device)

#         targets_x1 = target_fn1(data_x1)
#         targets_x2 = target_fn2(data_x2)

#         x1_z1, x1_logp1 = flow1.inverse(data_x1)
#         x1_x2, x1_logp2 = flow2.forward(x1_z1)
#         kl1 = (x1_logp1 - x1_logp2).mean()
#         mu1 = (x1_logp1 > x1_logp2).float().mean()

#         x2_z2, x2_logp2 = flow2.inverse(data_x2)
#         x2_x1, x2_logp1 = flow1.forward(x2_z2)
#         kl2 = (x2_logp2 - x2_logp1).mean()
#         mu2 = (x2_logp1 > x2_logp2).float().mean()

#         x1_out, x2_out = clf(x1_z1), clf(x2_z2)
#         x1_y, x2_y = x1_out.max(dim=1)[1], x2_out.max(dim=1)[1]
#         acc = 0.5 * ((x1_y == targets_x1).float().mean() + (x2_y == targets_x2).float().mean())
#         pred_loss = 0.5 * (F.cross_entropy(x1_out, targets_x1) + F.cross_entropy(x2_out, targets_x2))

#         if args.plot and epoch == args.n_epochs-1:
#             x1_z1 = x1_z1.detach().cpu().numpy()
#             x2_z2 = x2_z2.detach().cpu().numpy()

#             x1_colors = ['red' if targets_x1[i] == 0 else 'blue' for i in range(batch_size)]
#             x2_colors = ['red' if targets_x2[i] == 0 else 'blue' for i in range(batch_size)]
            
#             plt.scatter(x1_z1[:, 0], x1_z1[:, 1], color=x1_colors)
#             plt.scatter(x2_z2[:, 0], x2_z2[:, 1], color=x2_colors, marker='x')
#             plt.show()
#             exit(0)

#         if epoch % 10 == 0:
#             print('epoch: %d, kl1: %.3f, kl2: %.3f, mu1: %.3f, mu2: %.3f, pred_loss: %.3f, acc: %.3f' % (
#                   epoch, kl1.item(), kl2.item(), mu1, mu2, pred_loss.item(), acc.item()))
#         tot_loss = kl1 + kl2 + pred_loss
#         tot_loss.backward()
#         opt.step()

#     final_clf_acc = acc.item()

#     disc_samples = 2000
        
#     # Evaluating statistical distance
#     data_x1 = p1.sample((disc_samples,))
#     x1_z1, x1_logp1 = flow1.inverse(data_x1)
#     x1_x2, x1_logp2 = flow2.forward(x1_z1)
#     e1 = ((x1_logp1 - x1_logp2) > 0).float().mean()
    
#     data_x2 = p2.sample((disc_samples,))
#     x2_z2, x2_logp2 = flow2.inverse(data_x2)
#     x2_x1, x2_logp1 = flow1.forward(x2_z2)
#     e2 = ((x2_logp1 - x2_logp2) > 0).float().mean()

#     disc = torch.abs(e1-e2)
#     # print('stat discrepancy: ', disc)

#     # Training adversary
#     adv = nn.Sequential(
#         nn.Linear(d, 50),
#         nn.ReLU(),
#         nn.Linear(50, 50),
#         nn.ReLU(),
#         nn.Linear(50, 2),
#     )
#     adv = adv.to(device)
#     batch_size = 512
    
#     opt = optim.Adam(adv.parameters(), lr=1e-3)
#     lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.1)
    
#     for epoch in range(args.adv_epochs):
#         opt.zero_grad()

#         data_x1 = p1.sample((batch_size,))
#         data_x2 = p2.sample((batch_size,))

#         sens_x1 = torch.zeros(batch_size).long().to(device)
#         sens_x2 = torch.ones(batch_size).long().to(device)
        
#         x1_z1, x1_logp1 = flow1.inverse(data_x1)
#         x2_z2, x2_logp2 = flow2.inverse(data_x2)
        
#         x1_pred, x2_pred = adv(x1_z1), adv(x2_z2)
#         x1_y, x2_y = x1_pred.max(dim=1)[1], x2_pred.max(dim=1)[1]
#         x1_acc, x2_acc = (x1_y == sens_x1).float().mean(), (x2_y == sens_x2).float().mean()
#         loss = 0.5 * (F.cross_entropy(x1_pred, sens_x1) + F.cross_entropy(x2_pred, sens_x2))
#         loss.backward()
#         opt.step()
#         lr_scheduler.step()
#         # if epoch % 10 == 0:
#         #     print('adv: ' , epoch, loss.item(), 0.5 * (x1_acc + x2_acc).item())
#     print(loss.item(), 0.5 * (x1_acc + x2_acc).item(), final_clf_acc)


def get_gaussians(y1, y2):
    mean1, cov1 = torch.FloatTensor([0, y1]), torch.FloatTensor([[1, 0], [0, 1]])
    mean2, cov2 = torch.FloatTensor([0, y2]), torch.FloatTensor([[1, 0], [0, 1]])
    p1 = MultivariateNormal(mean1.to(device), cov1.to(device))
    p2 = MultivariateNormal(mean2.to(device), cov2.to(device))
    target_fn = lambda x: (x[:, 0] > 0).long()
    return 2, p1, p2, target_fn

def get_gaussians_flip(y1, y2):
    mean1, cov1 = torch.FloatTensor([0, y1]), torch.FloatTensor([[1, 0], [0, 1]])
    mean2, cov2 = torch.FloatTensor([0, y2]), torch.FloatTensor([[1, 0], [0, 1]])
    p1 = MultivariateNormal(mean1.to(device), cov1.to(device))
    p2 = MultivariateNormal(mean2.to(device), cov2.to(device))
    target_fn = lambda x: (x[:, 1].abs() < y1).long()
    return 2, p1, p2, target_fn


def get_stripes():
    
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
            g = torch.zeros((n, self.k)).to(device).to(device)
            g.scatter_(1, select, 1)
            samples = (g.unsqueeze(2) * pre_samples).sum(1)
            return samples

        def log_prob(self, r):
            # probs = []
            # for i in range(k):
            #     probs += [self.p[i].log_prob(r).exp().unsqueeze(0) * (1.0 / k)]
            # probs = torch.cat(probs, dim=0).sum(0)
            # print('-> ', probs.log())
            probs = []
            for i in range(self.k):
                probs += [self.p[i].log_prob(r).unsqueeze(0) + np.log(1.0 / self.k)]
            log_probs = torch.logsumexp(torch.cat(probs, dim=0), dim=0)
            return log_probs
            


    # p1 = GaussianMixture(4, [3, 6, 9, 12], [1, 1, 1, 1], 0.5)
    # p2 = GaussianMixture(4, [3, 6, 9, 12], [-1, -1, -1, -1], 0.5)
    # def target_fn1(r):
    #     return ((r[:, 0] < 4.5) | ((r[:, 0] >= 7.5) & (r[:, 0] < 10.5))).long()
    # def target_fn2(r):
    #     return (r[:, 0] < 7.5).long()

    p1 = GaussianMixture(2, [3, 6], [2, 2], 0.2)
    p2 = GaussianMixture(2, [3, 6], [-2, -2], 0.2)
    def target_fn1(r):
        return (r[:, 0] > 4.5).long()
    def target_fn2(r):
        return (r[:, 0] < 4.5).long()

    return 2, p1, p2, target_fn1, target_fn2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10, help='num of epochs')
    parser.add_argument('--adv_epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--n_blocks', type=int, default=10, help='num of blocks')
    parser.add_argument('--plot', action='store_true', help='whether to plot')
    parser.add_argument('--device', type=str, default='cpu', help='device to use')
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    global device
    device = args.device
    
    # d, p1, p2, target_fn = get_gaussians(3.5, -3.5)
    # d, p1, p2, target_fn1, target_fn2 = get_gaussians_flip(3.5, -3.5)
    d, p1, p2, target_fn1, target_fn2 = get_stripes()
    train_flow(args, d, p1, p2, target_fn1, target_fn2)
    


if __name__ == '__main__':
    main()
