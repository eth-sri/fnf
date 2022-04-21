import argparse
import random
from collections import defaultdict

import numpy as np
import torch.nn.functional as F
import torch.utils.data

import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--protected-att', type=str, default=None)
parser.add_argument('--p_test', type=float, default=0.2)
parser.add_argument('--p_val', type=float, default=0.2)
parser.add_argument('--load', action='store_true')
parser.add_argument('--label', type=str, default=None)
parser.add_argument('--transfer', action='store_true')
parser.add_argument('--quantiles', action='store_true')
parser.add_argument('--classifier-dims', type=int, nargs='*', required=True)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--with-test', action='store_true')
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = getattr(
    datasets, args.dataset.capitalize() + 'Dataset'
)('train', args, p_test=args.p_test, p_val=args.p_val, preprocess=True)
valid_dataset = getattr(
    datasets, args.dataset.capitalize() + 'Dataset'
)('validation', args, p_test=args.p_test, p_val=args.p_val, preprocess=True)
test_dataset = getattr(
    datasets, args.dataset.capitalize() + 'Dataset'
)('test', args, p_test=args.p_test, p_val=args.p_val, preprocess=True)

loaders = {
    split: torch.utils.data.DataLoader(
        split_dataset, batch_size=args.batch_size, shuffle=split == 'train'
    ) for split, split_dataset in {
        'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset
    }.items()
}

dims = [train_dataset.features.shape[1]] + args.classifier_dims + [2]
classifier = torch.nn.Sequential(*[
    torch.nn.Sequential(
        torch.nn.Linear(dim, dims[idx + 1]),
        torch.nn.ReLU() if idx < len(dims) - 2 else torch.nn.Identity()
    ) for idx, dim in enumerate(dims[:-1])
]).to(device)
optimizer = torch.optim.Adam(
    classifier.parameters(), lr=1e-3, weight_decay=args.weight_decay
)

metrics = defaultdict(dict)

for epoch in range(args.num_epochs + 1):
    if epoch < args.num_epochs:
        splits = ['train', 'valid']
    else:
        splits = ['valid'] if not args.with_test else ['valid', 'test']

    for split in splits:

        tot_acc1 = tot_loss1 = tot_samples1 = 0
        tot_acc2 = tot_loss2 = tot_samples2 = 0

        for inputs, targets, sens in loaders[split]:

            inputs, targets = inputs.to(device), targets.to(device).long()

            logits = classifier(inputs)
            loss1 = F.cross_entropy(logits[sens == 0], targets[sens == 0])
            loss2 = F.cross_entropy(logits[sens == 1], targets[sens == 1])

            tot_acc1 += (logits.argmax(1) == targets)[sens == 0].sum().item()
            tot_acc2 += (logits.argmax(1) == targets)[sens == 1].sum().item()
            tot_loss1 += loss1.sum().item()
            tot_loss2 += loss2.sum().item()
            tot_samples1 += torch.sum(sens == 0).item()
            tot_samples2 += torch.sum(sens == 1).item()

            if split == 'train':
                optimizer.zero_grad()
                loss = 0.5 * (loss1.mean() + loss2.mean())
                loss.backward()
                optimizer.step()

        unbal_acc = (tot_acc1 + tot_acc2) / (tot_samples1 + tot_samples2)
        bal_acc = 0.5 * ((tot_acc1 / tot_samples1) + (tot_acc2 / tot_samples2))
        loss = 0.5 * ((tot_loss1 / tot_samples1) + (tot_loss2 / tot_samples2))

        metrics[split]['balanced_acc'] = bal_acc
        metrics[split]['unbalanced_acc'] = unbal_acc
        metrics[split]['loss'] = loss

for split, split_metrics in metrics.items():
    for metric, value in split_metrics.items():
        print(f'[{split}]: {metric}={value:.5f}')
    print()
