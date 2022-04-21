import argparse
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset

from datasets.lawschool import LawschoolDataset
from generative.gmm import train_gmm
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
parser.add_argument('--gmm_comps1', type=int, default=2)
parser.add_argument('--gmm_comps2', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--out_file', type=str, default=None)
parser.add_argument('--n_flows', type=int, default=1)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--train_dec', action='store_true')
parser.add_argument('--log_epochs', type=int, default=10)
parser.add_argument('--quantiles', action='store_true')
parser.add_argument('--p_test', type=float, default=0.2)
parser.add_argument('--p_val', type=float, default=0.2)
parser.add_argument('--with_test', action='store_true')
parser.add_argument('--fair_criterion', type=str, default='stat_parity', choices=['stat_parity', 'eq_odds', 'eq_opp'])
parser.add_argument('--no_early_stop', action='store_true')
parser.add_argument('--load_enc', action='store_true')
parser.add_argument('--save_enc', action='store_true')
parser.add_argument('--save-encoding', action='store_true')
parser.add_argument('--scalarization', type=str, default='convex', choices=['convex', 'chebyshev'])
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = 'cuda'

train_dataset = LawschoolDataset('train', args, p_test=args.p_test, p_val=args.p_val)
valid_dataset = LawschoolDataset('validation', args, p_test=args.p_test, p_val=args.p_val)
test_dataset = LawschoolDataset('test', args, p_test=args.p_test, p_val=args.p_val)

train_all, train_prot, train_targets = train_dataset.features, train_dataset.protected, train_dataset.labels
valid_all, valid_prot, valid_targets = valid_dataset.features, valid_dataset.protected, valid_dataset.labels
test_all, test_prot, test_targets = test_dataset.features, test_dataset.protected, test_dataset.labels

train_college = train_all[:, 6:30].max(dim=1)[1]
valid_college = valid_all[:, 6:30].max(dim=1)[1]
test_college = test_all[:, 6:30].max(dim=1)[1]

c1_cnt = np.bincount(train_college[train_targets == 0].detach().cpu().numpy())
c1_cnt = c1_cnt / np.sum(c1_cnt)

college_rnk = list(range(c1_cnt.shape[0]))
college_rnk.sort(key=lambda i: c1_cnt[i])

new_train_college = train_college.detach().clone()
new_valid_college = valid_college.detach().clone()
new_test_college = test_college.detach().clone()

for i, college in enumerate(college_rnk):
    new_train_college = torch.where(train_college == college, i, new_train_college)
    new_valid_college = torch.where(valid_college == college, i, new_valid_college)
    new_test_college = torch.where(test_college == college, i, new_test_college)

train_all = torch.cat([train_all[:, :2], new_train_college.unsqueeze(1)], dim=1).float()
valid_all = torch.cat([valid_all[:, :2], new_valid_college.unsqueeze(1)], dim=1).float()
test_all = torch.cat([test_all[:, :2], new_test_college.unsqueeze(1)], dim=1).float()


def compute_quants(train_all):
    quants = []
    for i in range(train_all.shape[1]):
        x = np.sort(train_all[:, i].detach().cpu().numpy())
        min_quant = 1000.0
        for j in range(x.shape[0] - 1):
            if x[j+1] - x[j] < 1e-4:
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
    x = (1-args.alpha) * x + 0.5
    return torch.from_numpy(x).float().to(device), a, b


quants = compute_quants(train_all)
quants[1] = 0
for i in range(train_all.shape[1]):
    train_all[:, i], a, b = preprocess(i, train_all[:, i], quants[i])
    valid_all[:, i], _, _ = preprocess(i, valid_all[:, i], quants[i], a, b)
    test_all[:, i], _, _ = preprocess(i, test_all[:, i], quants[i], a, b)
q = torch.tensor(compute_quants(train_all)).float().unsqueeze(0).to(device)
q[0, 1] = 0

model = LogisticRegression()

ta = (train_all + q * torch.rand(train_all.shape).to(device)).logit()
va = (valid_all + q * torch.rand(valid_all.shape).to(device)).logit()

X = ta.detach().cpu().numpy()
y = train_targets.detach().cpu().numpy()
model.fit(X, y)
X_valid = va.detach().cpu().numpy()
y_valid = valid_targets.detach().cpu().numpy()

train1, train2 = train_all[train_prot == 1], train_all[train_prot == 0]
targets1, targets2 = train_targets[train_prot == 1].long(), train_targets[train_prot == 0].long()
train1_loader = torch.utils.data.DataLoader(TensorDataset(train1, targets1), batch_size=args.batch_size, shuffle=True, drop_last=False)
train2_loader = torch.utils.data.DataLoader(TensorDataset(train2, targets2), batch_size=args.batch_size, shuffle=True, drop_last=False)

valid1, valid2 = valid_all[valid_prot == 1], valid_all[valid_prot == 0]
v_targets1, v_targets2 = valid_targets[valid_prot == 1].long(), valid_targets[valid_prot == 0].long()
valid1_loader = torch.utils.data.DataLoader(TensorDataset(valid1, v_targets1), batch_size=args.batch_size, drop_last=False)
valid2_loader = torch.utils.data.DataLoader(TensorDataset(valid2, v_targets2), batch_size=args.batch_size, drop_last=False)

test1, test2 = test_all[test_prot == 1], test_all[test_prot == 0]
t_targets1, t_targets2 = test_targets[test_prot == 1].long(), test_targets[test_prot == 0].long()
test1_loader = torch.utils.data.DataLoader(TensorDataset(test1, t_targets1), batch_size=args.batch_size, drop_last=False)
test2_loader = torch.utils.data.DataLoader(TensorDataset(test2, t_targets2), batch_size=args.batch_size, drop_last=False)

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
prior1, prior2 = train_gmm(args, q, train, valid, device)

in_dim = train_all.shape[1]

train_loaders = (train1_loader, train2_loader)
valid_loaders = (valid1_loader, valid2_loader)
test_loaders = (test1_loader, test2_loader) if args.with_test else None
clf_dims = [100]
flow_dims = [50, 50]
flows = train_flow(args, in_dim, q, prior1, prior2, flow_dims, clf_dims, train_loaders, valid_loaders, test_loaders)

if args.save_encoding:
    model_dir = PROJECT_ROOT / 'code'/ 'lawschool' / f'gamma_{args.gamma}'
    model_dir.mkdir(parents=True, exist_ok=True)

    np.save(model_dir / 'prior1_weights', prior1[0].weights.cpu())
    np.save(model_dir / 'prior1_means', prior1[0].means.cpu())
    np.save(model_dir / 'prior1_covs', prior1[0].covs.cpu())

    np.save(model_dir / 'prior2_weights', prior2[0].weights.cpu())
    np.save(model_dir / 'prior2_means', prior2[0].means.cpu())
    np.save(model_dir / 'prior2_covs', prior2[0].covs.cpu())

    torch.save(flows[0][0].state_dict(), model_dir / 'flow1.pt')
    torch.save(flows[1][0].state_dict(), model_dir / 'flow2.pt')
