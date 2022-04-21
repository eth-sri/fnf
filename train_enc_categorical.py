import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from datasets.adult import AdultDataset
from datasets.compas import CompasDataset

PROJECT_ROOT = Path('.').absolute().parent

sns.set_theme()

parser = argparse.ArgumentParser()
parser.add_argument('--protected_att', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--adv_epochs', type=int, default=150)
parser.add_argument('--load_made', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--encode', action='store_true')
parser.add_argument('--dataset', type=str, required=True, choices=['compas', 'adult'])
parser.add_argument('--with_test', action='store_true')
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--out_file', type=str, default=None)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--p_test', type=float, default=0.2)
parser.add_argument('--p_val', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--save-encoding', action='store_true')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = args.device

if args.dataset == 'adult':
    train_dataset = AdultDataset('train', args, p_test=args.p_test, p_val=args.p_val)
    valid_dataset = AdultDataset('validation', args, p_test=args.p_test, p_val=args.p_val)
    test_dataset = AdultDataset('test', args, p_test=args.p_test, p_val=args.p_val)
elif args.dataset == 'compas':
    train_dataset = CompasDataset('train', args, p_test=args.p_test, p_val=args.p_val)
    valid_dataset = CompasDataset('validation', args, p_test=args.p_test, p_val=args.p_val)
    test_dataset = CompasDataset('test', args, p_test=args.p_test, p_val=args.p_val)
else:
    assert False, 'Unknown dataset'

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

# Split based on sensitive attribute
train_all, train_prot, train_labels = train_dataset.features, train_dataset.protected, train_dataset.labels
train1, train2 = train_all[train_prot == 1], train_all[train_prot == 0]
train_targets1, train_targets2 = train_labels[train_prot == 1], train_labels[train_prot == 0]

valid_all, valid_prot, valid_labels = valid_dataset.features, valid_dataset.protected, valid_dataset.labels
valid1, valid2 = valid_all[valid_prot == 1], valid_all[valid_prot == 0]
valid_targets1, valid_targets2 = valid_labels[valid_prot == 1], valid_labels[valid_prot == 0]

test_all, test_prot, test_labels = test_dataset.features, test_dataset.protected, test_dataset.labels
test1, test2 = test_all[test_prot == 1], test_all[test_prot == 0]
test_targets1, test_targets2 = test_labels[test_prot == 1], test_labels[test_prot == 0]

# Create loaders
train1_loader = torch.utils.data.DataLoader(train1, batch_size=args.batch_size, shuffle=True, drop_last=True)
train2_loader = torch.utils.data.DataLoader(train2, batch_size=args.batch_size, shuffle=True, drop_last=True)

valid1_loader = torch.utils.data.DataLoader(valid1, batch_size=args.batch_size, shuffle=False, drop_last=True)
valid2_loader = torch.utils.data.DataLoader(valid2, batch_size=args.batch_size, shuffle=False, drop_last=True)

test1_loader = torch.utils.data.DataLoader(test1, batch_size=args.batch_size, shuffle=False, drop_last=True)
test2_loader = torch.utils.data.DataLoader(test2, batch_size=args.batch_size, shuffle=False, drop_last=True)

dims = []
for ids in train_dataset.col_ids:
    dims += [len(ids)]
assert sum(dims) == train1.shape[1]

log_reg = LogisticRegression()
X, y = train_all.detach().cpu().numpy(), train_labels.detach().cpu().numpy()
X1, y1 = train1.detach().cpu().numpy(), train_targets1.detach().cpu().numpy()
X2, y2 = train2.detach().cpu().numpy(), train_targets2.detach().cpu().numpy()
w1, w2 = train1.shape[0]/train_all.shape[0], train2.shape[0]/train_all.shape[0]
sample_weights = np.where(train_prot.detach().cpu().numpy() == 0, w1, w2).reshape(-1)
log_reg.fit(X, y, sample_weights)
print('[1] log_reg: ', log_reg.score(X1, y1))
print('[2] log_reg: ', log_reg.score(X2, y2))

# Load priors
prior_made1 = torch.load('%s/made1.pt' % args.dataset, map_location=device)
prior_made2 = torch.load('%s/made2.pt' % args.dataset, map_location=device)

ids = [list(range(k)) for k in dims]

print('Base rate: SENS_1: %.4lf, SENS_2: %.4lf, ALL: %.4f' % (
      train_targets1.float().mean(), train_targets2.float().mean(),
      0.5 * (train_targets1.float().mean() + train_targets2.float().mean())))

if args.encode:
    with torch.no_grad():
        x = torch.zeros(np.prod(dims), sum(dims)).to(device)
        for idx, t in tqdm(enumerate(itertools.product(*ids))):
            beg = 0
            for k, i in zip(dims, t):
                x[idx, beg+i] = 1
                beg += k

        logp1, logp2 = prior_made1.log_prob(x), prior_made2.log_prob(x)
        # NOTE: Normalize as MADE can assign non-zero probability to something that is not one-hot
        logp1 = logp1 + (-logp1.exp().sum().log())
        logp2 = logp2 + (-logp2.exp().sum().log())
        logp_target = log_reg.predict_log_proba(x.detach().cpu().numpy())

        assert torch.abs(logp1.exp().sum() - 1) < 1e-5
        assert torch.abs(logp2.exp().sum() - 1) < 1e-5

        g_p, g_q = {0: [], 1: []}, {0: [], 1: []}
        p, q = [], []
        for idx, t in tqdm(enumerate(itertools.product(*ids))):
            p += [(logp1[idx].item(), t)]
            q += [(logp2[idx].item(), t)]
            g = np.argmax(logp_target[idx])
            g_p[g] += [(logp1[idx].item(), t)]
            g_q[g] += [(logp2[idx].item(), t)]

    p_z_1, p_z_2 = {}, {}
    tmap_group, tmap_all = {}, {}
    for g in [0, 1]:
        g_p[g].sort(key=lambda v: v[0])
        g_q[g].sort(key=lambda v: v[0])
        for (logp1, t1), (logp2, t2) in zip(g_p[g], g_q[g]):
            p_z_1[t1] = (1 - args.gamma) * np.exp(logp1)
            p_z_2[t1] = (1 - args.gamma) * np.exp(logp2)
            tmap_group[t2] = t1

    p.sort(key=lambda v: v[0])
    q.sort(key=lambda v: v[0])
    for (logp1, t1), (logp2, t2) in zip(p, q):
        p_z_1[t1] = p_z_1[t1] + args.gamma * np.exp(logp1)
        p_z_2[t1] = p_z_2[t1] + args.gamma * np.exp(logp2)
        tmap_all[t2] = t1

if args.encode:
    tot_p1, tot_p2 = 0, 0
    d = []
    for t in tqdm(itertools.product(*ids)):
        tot_p1 += p_z_1[t]
        tot_p2 += p_z_2[t]
        d += [p_z_1[t] - p_z_2[t]]
    d = np.array(d)
    mu1 = (d > 0.0).astype(float)
    mu2 = (d < 0.0).astype(float)
    stat_dist = max(np.abs(np.sum(d * mu1)), np.abs(np.sum(d * mu2)))
else:
    stat_dist = None

# classifier
clf = nn.Sequential(
    nn.Linear(sum(dims), 20),
    nn.ReLU(),
    nn.Linear(20, 2),
).to(device)
opt_clf = optim.Adam(clf.parameters(), lr=1e-3, weight_decay=args.weight_decay)


if args.encode:
    print('Encoding data...')

    modes = ['train', 'valid']
    if args.with_test:
        modes += ['test']

    encoded_data = {mode: defaultdict(list) for mode in modes}
    encoded_mapping = {
        mode: defaultdict(lambda: defaultdict(list)) for mode in modes
    }

    for mode in modes:
        if mode == 'train':
            data, sens = train_all, train_prot
        elif mode == 'valid':
            data, sens = valid_all, valid_prot
        elif mode == 'test':
            data, sens = test_all, test_prot

        for j in tqdm(range(data.shape[0])):

            t = []
            beg = 0
            for i, k in enumerate(dims):
                idx1 = data[j, beg:beg+k].max(dim=0)[1].item()
                t += [idx1]
                beg += k

            if args.save_encoding and (sens[j] == 1):
                encoded_mapping[mode][tuple(t)]['w'].append(j)

            if sens[j] == 1:
                continue

            data[j] = 0

            # Randomized encoding
            if np.random.rand() < 1 - args.gamma:
                w = tmap_group[tuple(t)]
            else:
                w = tmap_all[tuple(t)]

            beg = 0
            for i, k in enumerate(dims):
                data[j, beg+w[i]] = 1
                beg += k

            if args.save_encoding:
                t_array = np.zeros((1, sum(dims)))
                w_array = np.zeros((1, sum(dims)))

                beg = 0
                for idx, t_val in enumerate(t):
                    t_array[0, beg + t_val] = 1
                    beg += dims[idx]

                beg = 0
                for idx, w_val in enumerate(w):
                    w_array[0, beg + w_val] = 1
                    beg += dims[idx]

                t_y = log_reg.predict(t_array)
                w_y = log_reg.predict(w_array)

                encoded_mapping[mode][tuple(w)]['t'].append(j)
                encoded_mapping[mode][tuple(w)]['t_y'] = t_y
                encoded_mapping[mode][tuple(w)]['w_y'] = w_y

                encoded_data[mode]['t_x'].append(t_array[0])
                encoded_data[mode]['w_x'].append(w_array[0])
                encoded_data[mode]['t_y'].append(t_y)
                encoded_data[mode]['w_y'].append(w_y)
                encoded_data[mode]['t_a'].append(np.asarray([0]))
                encoded_data[mode]['w_a'].append(np.asarray([1]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.save_encoding:
        enc_dir = PROJECT_ROOT / 'encodings' / f'{args.dataset}_gamma_{args.gamma}'
        enc_dir.mkdir(parents=True, exist_ok=True)

        with open(enc_dir / 'column_ids_disc.json', 'w') as col_ids_file:
            json.dump(train_dataset.column_ids, col_ids_file)

        for mode, encoding in encoded_data.items():
            for enc, enc_data in encoding.items():
                np.savetxt(enc_dir / f'{mode}_{enc}.csv', np.vstack(enc_data))

        undiscretized_data = {
            'compas': {
                'train': CompasDataset('train', args, p_test=args.p_test, p_val=args.p_val, discretize=False),
                'valid': CompasDataset('validation', args, p_test=args.p_test, p_val=args.p_val, discretize=False),
                'test': CompasDataset('test', args, p_test=args.p_test, p_val=args.p_val, discretize=False)
            },
            'adult': {
                'train': AdultDataset('train', args, p_test=args.p_test, p_val=args.p_val, discretize=False),
                'valid': AdultDataset('validation', args, p_test=args.p_test, p_val=args.p_val, discretize=False),
                'test': AdultDataset('test', args, p_test=args.p_test, p_val=args.p_val, discretize=False)
            }
        }

        with open(enc_dir / 'column_ids_undisc.json', 'w') as col_ids_file:
            json.dump(undiscretized_data[args.dataset]['train'].column_ids, col_ids_file)

        for mode, enc_mapping in encoded_mapping.items():
            w_features, t_features = list(), list()
            w_labels, t_labels = list(), list()
            w_sizes, t_sizes = list(), list()

            for mapping in enc_mapping.values():
                if not mapping['w'] or not mapping['t']:
                    continue

                w_features.append(
                    torch.vstack([
                        undiscretized_data[args.dataset][mode].features[idx]
                        for idx in mapping['w']
                    ]).mean(0)
                )
                t_features.append(
                    torch.vstack([
                        undiscretized_data[args.dataset][mode].features[idx]
                        for idx in mapping['t']
                    ]).mean(0)
                )
                w_sizes.append(len(mapping['w']))
                t_sizes.append(len(mapping['t']))
                w_labels.append(mapping['w_y'])
                t_labels.append(mapping['t_y'])

            np.savetxt(enc_dir / f'{mode}_w_features.csv', torch.vstack(w_features).cpu())
            np.savetxt(enc_dir / f'{mode}_t_features.csv', torch.vstack(t_features).cpu())
            np.savetxt(enc_dir / f'{mode}_w_labels.csv', np.vstack(w_labels))
            np.savetxt(enc_dir / f'{mode}_t_labels.csv', np.vstack(t_labels))
            np.savetxt(enc_dir / f'{mode}_w_sizes.csv', w_sizes)
            np.savetxt(enc_dir / f'{mode}_t_sizes.csv', t_sizes)

        exit()

test_unbal_acc, test_bal_acc = -1, -1
valid_unbal_acc, valid_bal_acc = -1, -1

print('Training classifier...')
for epoch in range(args.n_epochs+1):
    if epoch == args.n_epochs:
        modes = ['valid']
        if args.with_test:
            modes += ['test']
    else:
        modes = ['train', 'valid']

    for mode in modes:
        tot_data_acc1, tot_data_loss1, tot_samples1 = 0, 0, 0
        tot_data_acc2, tot_data_loss2, tot_samples2 = 0, 0, 0
        if mode == 'train':
            data_loader = train_loader
        elif mode == 'valid':
            data_loader = valid_loader
        elif mode == 'test':
            data_loader = test_loader
        else:
            assert False
        for inputs, targets, sens in data_loader:
            inputs, targets = inputs.to(device), targets.to(device).long()
            tot_samples1 += (sens == 0).float().sum().item()
            tot_samples2 += (sens == 1).float().sum().item()
            outs = clf(inputs)

            data_loss1 = F.cross_entropy(outs[sens == 0], targets[sens == 0])
            data_loss2 = F.cross_entropy(outs[sens == 1], targets[sens == 1])
            data_acc1 = outs[sens == 0].max(dim=1)[1].eq(targets[sens == 0]).float()
            data_acc2 = outs[sens == 1].max(dim=1)[1].eq(targets[sens == 1]).float()

            tot_data_loss1 += data_loss1.sum().item()
            tot_data_loss2 += data_loss2.sum().item()
            tot_data_acc1 += data_acc1.sum().item()
            tot_data_acc2 += data_acc2.sum().item()

            if mode == 'train':
                opt_clf.zero_grad()
                loss = 0.5*(data_loss1.mean() + data_loss2.mean())
                loss.backward()
                opt_clf.step()
        data_unbal_acc = ((tot_data_acc1+tot_data_acc2)/(tot_samples1+tot_samples2))
        data_bal_acc = 0.5*(tot_data_acc1/tot_samples1 + tot_data_acc2/tot_samples2)
        data_loss = 0.5*(tot_data_loss1/tot_samples1 + tot_data_loss2/tot_samples2)
        if args.verbose and (epoch == args.n_epochs or (epoch+1) % 10 == 0):
            print('clf [%s] epoch: %d, loss: %.4f, unbal_acc: %.4f, bal_acc: %.4f' % (
                mode, epoch+1, data_loss, data_unbal_acc, data_bal_acc))
        if epoch == args.n_epochs:
            if mode == 'valid':
                valid_unbal_acc, valid_bal_acc = data_unbal_acc, data_bal_acc
            if mode == 'test':
                test_unbal_acc, test_bal_acc = data_unbal_acc, data_bal_acc

print('Statistical distance: ', stat_dist)
print('Final valid accuracy: ', valid_bal_acc)
print('Final test accuracy: ', test_bal_acc)

def train_adversary(adv_epochs, lr, weight_decay, n_hidden, hidden_dim, with_test=False):
    layers = [nn.Linear(sum(dims), hidden_dim)]
    for i in range(n_hidden-1):
        layers += [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
    layers += [nn.Linear(hidden_dim, 2)]
    adv = nn.Sequential(*layers).to(device)
    opt_adv = optim.Adam(adv.parameters(), lr=lr, weight_decay=weight_decay)

    final_valid_acc, final_test_acc = -1, -1
    for epoch in range(adv_epochs):
        modes = ['train', 'valid']
        if with_test:
            modes += ['test']
        for mode in modes:
            tot_data_acc1, tot_data_loss1, tot_samples1 = 0, 0, 0
            tot_data_acc2, tot_data_loss2, tot_samples2 = 0, 0, 0
            if mode == 'train':
                data_loader = train_loader
            elif mode == 'valid':
                data_loader = valid_loader
            elif mode == 'test':
                data_loader = test_loader
            else:
                assert False
            for inputs, _, sens in data_loader:
                tot_samples1 += (sens == 0).float().sum().item()
                tot_samples2 += (sens == 1).float().sum().item()

                outs = adv(inputs)
                data_loss1 = F.cross_entropy(outs[sens == 1], sens[sens == 1])
                data_loss2 = F.cross_entropy(outs[sens == 0], sens[sens == 0])
                data_acc1 = outs[sens == 0].max(dim=1)[1].eq(sens[sens == 0]).float()
                data_acc2 = outs[sens == 1].max(dim=1)[1].eq(sens[sens == 1]).float()

                tot_data_loss1 += data_loss1.sum().item()
                tot_data_loss2 += data_loss2.sum().item()
                tot_data_acc1 += data_acc1.sum().item()
                tot_data_acc2 += data_acc2.sum().item()
                if mode == 'train':
                    opt_adv.zero_grad()
                    loss = 0.5*(data_loss1.mean() + data_loss2.mean())
                    loss.backward()
                    opt_adv.step()
            data_loss = 0.5*(tot_data_loss1/tot_samples1 + tot_data_loss2/tot_samples2)
            data_acc = 0.5*(tot_data_acc1/tot_samples1 + tot_data_acc2/tot_samples2)
            if args.verbose and (epoch+1) % 10 == 0:
                if mode == 'train':
                    print('')
                print('adv [%s] epoch: %d, tot_loss: %.4f, tot_acc: %.4f' % (
                    mode, epoch+1, data_loss, data_acc))

            if epoch == adv_epochs-1:
                if mode == 'valid':
                    final_valid_acc = data_acc
                if mode == 'test':
                    final_test_acc = data_acc
    return final_valid_acc, final_test_acc


best_config, best_adv_valid_acc, best_adv_test_acc = -1, -1, -1

for adv_epochs in [60]: #[10, 30, 60]:
    for adv_lr in [1e-2]: #[1e-2, 1e-3]:
        for adv_weight_decay in [1e-3]: #[1e-3, 1e-4]:
            for n_hidden in [2]:
                for hidden_dim in [30]:
                    adv_valid_acc, adv_test_acc = train_adversary(adv_epochs, adv_lr, adv_weight_decay, n_hidden, hidden_dim, args.with_test)
                    if adv_valid_acc > best_adv_valid_acc:
                        best_adv_valid_acc = adv_valid_acc
                        best_adv_test_acc = adv_test_acc
                        best_config = (adv_epochs, adv_lr, adv_weight_decay, n_hidden, hidden_dim, args.with_test)
                    print(adv_epochs, adv_lr, adv_weight_decay, n_hidden, hidden_dim, ' ---> ', adv_valid_acc, adv_test_acc)
print('Adversary:')
print(best_config)
print(best_adv_valid_acc)
print(best_adv_test_acc)


if args.out_file is not None:
    print('saving to: ', args.out_file)
    with open(args.out_file, 'a') as csvfile:
        field_names = ['gamma', 'stat_dist', 'valid_unbal_acc', 'valid_bal_acc', 'test_unbal_acc', 'test_bal_acc', 'adv_valid_acc', 'adv_test_acc']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writerow({'gamma': args.gamma, 'stat_dist': stat_dist,
                         'valid_unbal_acc': valid_unbal_acc, 'valid_bal_acc': valid_bal_acc,
                         'test_unbal_acc': test_unbal_acc, 'test_bal_acc': test_bal_acc,
                         'adv_valid_acc': adv_valid_acc, 'adv_test_acc': adv_test_acc})
