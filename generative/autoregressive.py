import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from tqdm import tqdm

# Wrapper for auto-regressive model so that it has same interface as RealNVP
class AutoReg(nn.Module):

    def __init__(self, k, device):
        super(AutoReg, self).__init__()
        self.k = k
        self.device = device
        self.layers = [
            nn.Sequential(
                nn.Linear(i+1, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 2)
            )
            for i in range(k)
        ]
        self.layers = nn.ModuleList(self.layers)
        self.norm = D.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))

    def sample(self, shape):
        n = shape[0]
        x = torch.zeros((n, 1)).to(self.device)
        for i in range(self.k):
            outs = self.layers[i](x)
            mu, log_var = outs[:, 0:1], outs[:, 1:2]
            z = self.norm.sample((n,))
            tmp = mu + z * (log_var / 2).exp()
            x = torch.cat([x, tmp], dim=1)
        return x[:, 1:]

    def log_p(self, x):
        log_p = 0
        for i in range(self.k):
            t = torch.zeros((x.shape[0], i+1)).to(x.device)
            if i > 0:
                t[:, 1:] = x[:, :i]
            outs = self.layers[i](t)
            mu, log_var = outs[:, 0], outs[:, 1]
            z = (x[:, i] - mu) / (log_var / 2).exp()
            log_p = log_p + self.norm.log_prob(z) - (log_var / 2)
        return log_p

    def log_prob(self, x):
        return self.log_p(x)


def train_autoreg(args, q, train, train_loaders, valid, device):
    train1_loader, train2_loader = train_loaders
    train1, train2, targets1, targets2 = train
    valid1, valid2, v_targets1, v_targets2 = valid
    in_dim = train1.shape[1]

    prior1 = AutoReg(train1.shape[1], device).to(device)
    prior2 = AutoReg(train1.shape[1], device).to(device)

    opt1 = optim.Adam(prior1.parameters(), lr=1e-2, weight_decay=1e-4)
    opt2 = optim.Adam(prior2.parameters(), lr=1e-2, weight_decay=1e-4)
    lr_scheduler1 = optim.lr_scheduler.MultiStepLR(opt1, milestones=[args.prior_epochs//3, 2*args.prior_epochs//3], gamma=0.1)
    lr_scheduler2 = optim.lr_scheduler.MultiStepLR(opt2, milestones=[args.prior_epochs//3, 2*args.prior_epochs//3], gamma=0.1)
    
    v1 = torch.clamp(valid1 + q * torch.rand(valid1.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
    v2 = torch.clamp(valid2 + q * torch.rand(valid2.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
    
    for epoch in range(args.prior_epochs):
        tot_loss1, tot_loss2, n_batches = 0, 0, 0
        for (inputs1, targets1), (inputs2, targets2) in zip(train1_loader, train2_loader):
            if q is not None:
                t1 = torch.clamp(inputs1 + q * torch.rand(inputs1.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
                t2 = torch.clamp(inputs2 + q * torch.rand(inputs2.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
            else:
                t1 = inputs1
                t2 = inputs2

            opt1.zero_grad()
            opt2.zero_grad()
            logp1 = prior1.log_p(t1)
            logp2 = prior2.log_p(t2)
            loss1 = -logp1.mean()
            loss2 = -logp2.mean()
            loss = loss1 + loss2
            tot_loss1 += loss1.item()
            tot_loss2 += loss2.item()
            n_batches += 1
            loss.backward()
            opt1.step()
            opt2.step()
        lr_scheduler1.step()
        lr_scheduler2.step()
        if (epoch+1) % 10 == 0:
            print(epoch+1, tot_loss1/n_batches, tot_loss2/n_batches)
            print('-> ', -prior1.log_p(v1).mean())
            print('-> ', -prior2.log_p(v2).mean())
            print('')

    for parameter in prior1.parameters():
        parameter.requires_grad_(False)
    for parameter in prior2.parameters():
        parameter.requires_grad_(False)
        
    return [prior1], [prior2]
