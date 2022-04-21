import argparse
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from datasets.crime import CrimeDataset
from generative.autoregressive import train_autoreg
from generative.flow import train_flow_prior
from generative.gmm import train_gmm
from real_nvp_encoder import FlowEncoder
from train_fnf import train_flow

PROJECT_ROOT = Path('.').absolute().parent

sns.set_theme()

device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--kl_start', type=int, default=50)
parser.add_argument('--kl_end', type=int, default=600)
parser.add_argument('--protected_att', type=str, default=None)
parser.add_argument('--n_blocks', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dec_epochs', type=int, default=100)
parser.add_argument('--prior_epochs', type=int, default=150)
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--adv_epochs', type=int, default=150)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--prior', type=str, default='flow', choices=['flow', 'gmm', 'autoreg'])
parser.add_argument('--gmm_comps1', type=int, default=4)
parser.add_argument('--gmm_comps2', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--out_file', type=str, default=None)
parser.add_argument('--n_flows', type=int, default=1)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--train_dec', action='store_true')
parser.add_argument('--log_epochs', type=int, default=10)
parser.add_argument('--p_test', type=float, default=0.2)
parser.add_argument('--p_val', type=float, default=0.2)
parser.add_argument('--with_test', action='store_true')
parser.add_argument('--fair_criterion', type=str, default='stat_parity', choices=['stat_parity', 'eq_odds', 'eq_opp'])
parser.add_argument('--load_prior', action='store_true')
parser.add_argument('--load_enc', action='store_true')
parser.add_argument('--no_early_stop', action='store_true')
parser.add_argument('--save_enc', action='store_true')
parser.add_argument('--save-encoding', action='store_true')
parser.add_argument('--scalarization', type=str, default='convex', choices=['convex', 'chebyshev'])
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = args.device

train_dataset = CrimeDataset('train', args, p_test=args.p_test, p_val=args.p_val)
valid_dataset = CrimeDataset('validation', args, p_test=args.p_test, p_val=args.p_val)
test_dataset = CrimeDataset('test', args, p_test=args.p_test, p_val=args.p_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

train_all, train_prot, train_targets = train_dataset.features, train_dataset.protected, train_dataset.labels
valid_all, valid_prot, valid_targets = valid_dataset.features, valid_dataset.protected, valid_dataset.labels
test_all, test_prot, test_targets = test_dataset.features, test_dataset.protected, test_dataset.labels

# Preprocessing
feats = np.array([3, 15, 42, 43, 44, 49])
quants = []
for i in feats:
    a, b = torch.min(train_all[:, i]), torch.max(train_all[:, i])
    train_all[:, i] = 0.5 + (1-args.alpha)*((train_all[:, i] - a) / (b - a) - 0.5)
    valid_all[:, i] = 0.5 + (1-args.alpha)*((valid_all[:, i] - a) / (b - a) - 0.5)
    test_all[:, i] = 0.5 + (1-args.alpha)*((test_all[:, i] - a) / (b - a) - 0.5)
    quants += [0]

q = torch.tensor(quants).float().unsqueeze(0).to(device)
train_all = train_all[:, feats]
valid_all = valid_all[:, feats]
test_all = test_all[:, feats]

train1, train2 = train_all[train_prot == 1], train_all[train_prot == 0]
targets1, targets2 = train_targets[train_prot == 1].long(), train_targets[train_prot == 0].long()
train1_loader = torch.utils.data.DataLoader(TensorDataset(train1, targets1), batch_size=args.batch_size, shuffle=True)
train2_loader = torch.utils.data.DataLoader(TensorDataset(train2, targets2), batch_size=args.batch_size, shuffle=True)

valid1, valid2 = valid_all[valid_prot == 1], valid_all[valid_prot == 0]
v_targets1, v_targets2 = valid_targets[valid_prot == 1].long(), valid_targets[valid_prot == 0].long()
valid1_loader = torch.utils.data.DataLoader(TensorDataset(valid1, v_targets1), batch_size=8, shuffle=True)
valid2_loader = torch.utils.data.DataLoader(TensorDataset(valid2, v_targets2), batch_size=8, shuffle=True)

test1, test2 = test_all[test_prot == 1], test_all[test_prot == 0]
t_targets1, t_targets2 = test_targets[test_prot == 1].long(), test_targets[test_prot == 0].long()
test1_loader = torch.utils.data.DataLoader(TensorDataset(test1, t_targets1), batch_size=8, shuffle=True)
test2_loader = torch.utils.data.DataLoader(TensorDataset(test2, t_targets2), batch_size=8, shuffle=True)

train_loaders = (train1_loader, train2_loader)
valid_loaders = (valid1_loader, valid2_loader)
test_loaders = (test1_loader, test2_loader) if args.with_test else None

print('Base rates:')
print('p(y=1|a=1) = %.3f, p(y=1|a=0) = %.3f, p(y=1) = %.3f' % (
      targets1.float().mean(), targets2.float().mean(),
      0.5 * (targets1.float().mean() + targets2.float().mean())))
print('p(a=1) = %.3f, p(a=0) = %.3f' % (
    train1.shape[0]/float(train1.shape[0] + train2.shape[0]), train2.shape[0]/float(train1.shape[0] + train2.shape[0])))
for y in range(2):
    a0 = (targets1 == y).float().sum()
    a1 = (targets2 == y).float().sum()
    print('p(a=1|y=%d) = %.3f, p(a=0|y=%d) = %.3f' % (y, a0/(a0 + a1), y, a1/(a0 + a1)))

in_dim = feats.shape[0]

if args.prior == 'flow':
    train = (train1, train2, targets1, targets2)
    valid = (valid1, valid2, v_targets1, v_targets2)
    prior1, prior2 = train_flow_prior(args, q, train, train_loaders, valid, device)
elif args.prior == 'gmm':
    train = (train1, train2, targets1, targets2)
    valid = (valid1, valid2, v_targets1, v_targets2)
    prior1, prior2 = train_gmm(args, q, train, valid, device)
elif args.prior == 'autoreg':
    train = (train1, train2, targets1, targets2)
    valid = (valid1, valid2, v_targets1, v_targets2)
    prior1, prior2 = train_autoreg(args, q, train, train_loaders, valid, device)
else:
    assert False, 'Unknown prior!'


def predict(inputs, targets, dec_net, flows):
    alphas = F.softmax(dec_net(inputs), dim=1)
    acc, pred_loss = 0, 0
    for i in range(len(flows)):
        z = flows[i].inverse(inputs)[0]
        out = clf(z)
        y = out.max(dim=1)[1]
        acc = acc + alphas[:, i] * (y == targets).float()
        pred_loss = pred_loss + alphas[:, i] * F.cross_entropy(out, targets)
    return acc, pred_loss

    
n_flows = args.n_flows
flows = [[FlowEncoder(None, in_dim, [20], 4).to(device) for i in range(n_flows)] for _ in range(2)]
for i in range(n_flows):
    flows[1][i].load_state_dict(flows[0][i].state_dict())

clf = nn.Sequential(
    nn.Linear(in_dim, 100),
    nn.ReLU(),
    nn.Linear(100, 2),
).to(device)

dec_nets = [
    nn.Sequential(
        nn.Linear(in_dim, 20),
        nn.ReLU(),
        nn.Linear(20, n_flows),
    ).to(device)
    for _ in range(2)]

dec_params = []
for d in dec_nets:
    dec_params += list(d.parameters())

clf_dims = [100]
flow_dims = [50, 50]

if args.scalarization == 'convex':
    flows = train_flow(args, in_dim, q, prior1, prior2, flow_dims, clf_dims, train_loaders, valid_loaders, test_loaders)

elif args.scalarization == 'chebyshev':

    gamma = args.gamma

    args.gamma = 0
    _, (min_pred_loss, max_pred_loss), _ = train_flow(
        args, in_dim, q, prior1, prior2, flow_dims, clf_dims, train_loaders, valid_loaders, test_loaders, return_loss_bounds=True
    )

    args.gamma = 1
    _, _, (min_kl_loss, max_kl_loss) = train_flow(
        args, in_dim, q, prior1, prior2, flow_dims, clf_dims, train_loaders, valid_loaders, test_loaders, return_loss_bounds=True
    )

    args.gamma = gamma
    flows = train_flow(
        args, in_dim, q, prior1, prior2, flow_dims, clf_dims, train_loaders, valid_loaders, test_loaders,
        lb_pred_loss=min_pred_loss, ub_pred_loss=max_pred_loss, lb_kl_loss=min_kl_loss, ub_kl_loss=max_kl_loss
    )

if args.save_encoding:

    model_dir = PROJECT_ROOT / 'code'/ 'crime' / f'gamma_{args.gamma}'
    model_dir.mkdir(parents=True, exist_ok=True)

    np.save(model_dir / 'prior1_weights', prior1[0].weights.cpu())
    np.save(model_dir / 'prior1_means', prior1[0].means.cpu())
    np.save(model_dir / 'prior1_covs', prior1[0].covs.cpu())

    np.save(model_dir / 'prior2_weights', prior2[0].weights.cpu())
    np.save(model_dir / 'prior2_means', prior2[0].means.cpu())
    np.save(model_dir / 'prior2_covs', prior2[0].covs.cpu())

    torch.save(flows[0][0].state_dict(), model_dir / 'flow1.pt')
    torch.save(flows[1][0].state_dict(), model_dir / 'flow2.pt')
