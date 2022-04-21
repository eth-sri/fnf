import argparse
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets.adult import AdultDataset
from datasets.compas import CompasDataset
from generative.made import MADE

sns.set_theme()

device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--protected_att', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--hidden', type=int, default=50)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dataset', type=str, required=True, choices=['compas', 'adult'])
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

device = args.device

if args.dataset == 'adult':
    train_dataset = AdultDataset('train', args)
    valid_dataset = AdultDataset('validation', args)
elif args.dataset == 'compas':
    train_dataset = CompasDataset('train', args)
    valid_dataset = CompasDataset('validation', args)
else:
    assert False, 'Unknown dataset'

train_all = train_dataset.features
train_prot = train_dataset.protected
train1 = train_all[train_prot == 1]
train2 = train_all[train_prot == 0]

valid_all = valid_dataset.features
valid_prot = valid_dataset.protected
valid1 = valid_all[valid_prot == 1]
valid2 = valid_all[valid_prot == 0]

train1_loader = torch.utils.data.DataLoader(train1, batch_size=args.batch_size, shuffle=True)
train2_loader = torch.utils.data.DataLoader(train2, batch_size=args.batch_size, shuffle=True)

valid1_loader = torch.utils.data.DataLoader(valid1, batch_size=args.batch_size, shuffle=True)
valid2_loader = torch.utils.data.DataLoader(valid2, batch_size=args.batch_size, shuffle=True)

dims = [len(ids) for ids in train_dataset.col_ids]

made1 = MADE(sum(dims), [args.hidden, args.hidden]).to(device)
made2 = MADE(sum(dims), [args.hidden, args.hidden]).to(device)

opt_made1 = optim.Adam(list(made1.parameters()), lr=1e-2, weight_decay=1e-4)
opt_made2 = optim.Adam(list(made2.parameters()), lr=1e-2, weight_decay=1e-4)

lr_scheduler1 = optim.lr_scheduler.StepLR(opt_made1, step_size=args.n_epochs//2, gamma=0.1)
lr_scheduler2 = optim.lr_scheduler.StepLR(opt_made2, step_size=args.n_epochs//2, gamma=0.1)

best_valid_loss1, best_valid_loss2 = None, None

for epoch in range(args.n_epochs):
    tot_loss1, n_batches1 = 0, 0
    for inputs1 in train1_loader:
        opt_made1.zero_grad()
        n_batches1 += 1
        inputs1 = inputs1.to(device)
        outs1 = made1(inputs1)
        loss = F.binary_cross_entropy_with_logits(outs1, inputs1, reduction='none')
        loss = loss.sum(-1).mean()
        loss.backward()
        opt_made1.step()
        tot_loss1 += loss.item()

    tot_loss2, n_batches2 = 0, 0
    for inputs2 in train2_loader:
        opt_made2.zero_grad()
        n_batches2 += 1
        inputs2 = inputs2.to(device)
        outs2 = made2(inputs2)
        loss = F.binary_cross_entropy_with_logits(outs2, inputs2, reduction='none')
        loss = loss.sum(-1).mean()
        loss.backward()
        opt_made2.step()
        tot_loss2 += loss.item()

    if (epoch+1) % 20 == 0:
        print('epoch: %d, loss1: %.4f, loss2: %.4f' % (epoch+1, tot_loss1/n_batches1, tot_loss2/n_batches2))

    with torch.no_grad():
        tot_loss1, tot_loss2, n_batches = 0, 0, 0

        inputs1 = valid1.to(device)
        outs1 = made1(inputs1)
        loss = F.binary_cross_entropy_with_logits(outs1, inputs1, reduction='none')
        valid_loss1 = loss.sum(-1).mean()

        inputs2 = valid2.to(device)
        outs2 = made2(inputs2)
        loss = F.binary_cross_entropy_with_logits(outs2, inputs2, reduction='none')
        valid_loss2 = loss.sum(-1).mean().item()

        if (epoch + 1) % 20 == 0:
            print('[valid] epoch: %d, loss1: %.4f, loss2: %.4f' % (epoch+1, valid_loss1, valid_loss2))
        if best_valid_loss1 is None or valid_loss1 < best_valid_loss1:
            if args.verbose:
                print('best valid_loss_1, saving network...')
            best_valid_loss1 = valid_loss1
            torch.save(made1, '%s/made1.pt' % (args.dataset))
        if best_valid_loss2 is None or valid_loss2 < best_valid_loss2:
            if args.verbose:
                print('best valid_loss_2, saving network...')
            best_valid_loss2 = valid_loss2
            torch.save(made2, '%s/made2.pt' % (args.dataset))

    lr_scheduler1.step()
    lr_scheduler2.step()
