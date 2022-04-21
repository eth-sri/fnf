import argparse
import numpy as np
import torch
from datasets.health import HealthDataset
from torch.utils.data import TensorDataset
from generative.autoregressive import train_autoreg
from generative.gmm import train_gmm
from generative.flow import train_flow_prior
from train_fnf import train_flow
from sklearn.linear_model import LogisticRegression

device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--prior', type=str, default='flow', choices=['flow', 'gmm', 'autoreg'])
parser.add_argument('--kl_start', type=int, default=50)
parser.add_argument('--kl_end', type=int, default=150)
parser.add_argument('--protected_att', type=str, default=None)
parser.add_argument('--n_blocks', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dec_epochs', type=int, default=100)
parser.add_argument('--prior_epochs', type=int, default=150)
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--adv_epochs', type=int, default=150)
parser.add_argument('--load_made', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--label', type=str, default=None)
parser.add_argument('--load', action='store_true')
parser.add_argument('--transfer', action='store_true')
parser.add_argument('--load_prior', action='store_true')
parser.add_argument('--load_enc', action='store_true')
parser.add_argument('--save_enc', action='store_true')
parser.add_argument('--load_clf', action='store_true')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--out_file', type=str, default=None)
parser.add_argument('--with_test', action='store_true')
parser.add_argument('--eq_opp', type=int, default=None)
parser.add_argument('--n_flows', type=int, default=1)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--train_dec', action='store_true')
parser.add_argument('--log_epochs', type=int, default=10)
parser.add_argument('--p_test', type=float, default=0.2)
parser.add_argument('--p_val', type=float, default=0.2)
parser.add_argument('--gmm_comps1', type=int, default=2)
parser.add_argument('--gmm_comps2', type=int, default=2)
parser.add_argument('--fair_criterion', type=str, default='stat_parity', choices=['stat_parity', 'eq_odds', 'eq_opp'])
parser.add_argument('--schedule', action='store_true')
parser.add_argument('--no_early_stop', action='store_true')
parser.add_argument('--scalarization', type=str, default='convex', choices=['convex', 'chebyshev'])
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


device = args.device

train_dataset = HealthDataset('train', args, p_test=args.p_test, p_val=args.p_val)
valid_dataset = HealthDataset('validation', args, p_test=args.p_test, p_val=args.p_val)
test_dataset = HealthDataset('test', args, p_test=args.p_test, p_val=args.p_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

train_all, train_prot, train_targets = train_dataset.features, train_dataset.protected, train_dataset.labels
valid_all, valid_prot, valid_targets = valid_dataset.features, valid_dataset.protected, valid_dataset.labels
test_all, test_prot, test_targets = test_dataset.features, test_dataset.protected, test_dataset.labels

print(train_all.shape, valid_all.shape, test_all.shape)

if args.transfer:
    feats = [0, 1, 2, 3, 16, 31, 44]
else:
    feats = [2, 3, 4, 5, 8, 19, 33, 61, 71, 77, 92]

in_dim = len(feats)


def compute_quants(train_all):
    quants = []
    for i in range(train_all.shape[1]):
        x = np.sort(train_all[:, i].detach().cpu().numpy())
        min_quant = 1000.0
        for j in range(x.shape[0] - 1):
            if x[j+1] == x[j]:
                continue
            min_quant = min(min_quant, x[j+1] - x[j])
        quants += [min_quant]
    return quants


def preprocess(i, x, min_quant, a=None, b=None):
    x = x.detach().cpu().numpy()
    if a is None:
        a, b = np.min(x), np.max(x) + min_quant
    x = np.clip(x, a, b)
    x = (x - a) / (b - a) - 0.5
    x = (1 - args.alpha) * x + 0.5
    return torch.from_numpy(x).float().to(device), a, b


quants = compute_quants(train_all)
for i in range(train_all.shape[1]):
    train_all[:, i], a, b = preprocess(i, train_all[:, i], quants[i])
    valid_all[:, i], _, _ = preprocess(i, valid_all[:, i], quants[i], a, b)
    test_all[:, i], _, _ = preprocess(i, test_all[:, i], quants[i], a, b)
quants = compute_quants(train_all)
q = torch.tensor(quants).float().unsqueeze(0).to(device)
q = q[:, feats]

train_all = train_all[:, feats]
valid_all = valid_all[:, feats]
test_all = test_all[:, feats]

train1, train2 = train_all[train_prot == 1], train_all[train_prot == 0]
targets1, targets2 = train_targets[train_prot == 1].long(), train_targets[train_prot == 0].long()
train1_loader = torch.utils.data.DataLoader(TensorDataset(train1, targets1), batch_size=args.batch_size, shuffle=True, drop_last=True)
train2_loader = torch.utils.data.DataLoader(TensorDataset(train2, targets2), batch_size=args.batch_size, shuffle=True, drop_last=True)

valid1, valid2 = valid_all[valid_prot == 1], valid_all[valid_prot == 0]
v_targets1, v_targets2 = valid_targets[valid_prot == 1].long(), valid_targets[valid_prot == 0].long()
valid1_loader = torch.utils.data.DataLoader(TensorDataset(valid1, v_targets1), batch_size=args.batch_size, shuffle=True, drop_last=False)
valid2_loader = torch.utils.data.DataLoader(TensorDataset(valid2, v_targets2), batch_size=args.batch_size, shuffle=True, drop_last=False)

test1, test2 = test_all[test_prot == 1], test_all[test_prot == 0]
t_targets1, t_targets2 = test_targets[test_prot == 1].long(), test_targets[test_prot == 0].long()
test1_loader = torch.utils.data.DataLoader(TensorDataset(test1, t_targets1), batch_size=args.batch_size, shuffle=True, drop_last=False)
test2_loader = torch.utils.data.DataLoader(TensorDataset(test2, t_targets2), batch_size=args.batch_size, shuffle=True, drop_last=False)

print('Base rates:')
print('p(y=1|a=0) = %.3f, p(y=1|a=0) = %.3f, p(y=1) = %.3f' % (
      targets1.float().mean(), targets2.float().mean(),
      0.5 * (targets1.float().mean() + targets2.float().mean())))
print('p(a=0) = %.3f, p(a=1) = %.3f' % (
    train1.shape[0]/float(train1.shape[0] + train2.shape[0]), train2.shape[0]/float(train1.shape[0] + train2.shape[0])))
for y in range(2):
    a0 = (targets1 == y).float().sum()
    a1 = (targets2 == y).float().sum()
    print('p(a=0|y=%d) = %.3f, p(a=1|y=%d) = %.3f' % (y, a0/(a0 + a1), y, a1/(a0 + a1)))

train = (train1, train2, targets1, targets2)
valid = (valid1, valid2, v_targets1, v_targets2)
train_loaders = (train1_loader, train2_loader)
valid_loaders = (valid1_loader, valid2_loader)
test_loaders = (test1_loader, test2_loader) if args.with_test else None

prior1, prior2 = train_flow_prior(args, q, train, train_loaders, valid, device)

clf_dims = [100, 50, 20]
flow_dims = [100, 100, 50]
train_flow(args, in_dim, q, prior1, prior2, flow_dims, clf_dims, train_loaders, valid_loaders, test_loaders)
