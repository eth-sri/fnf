import argparse
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from toy_fnf import get_stripes

sns.set_theme()

device = 'cuda'
ce_loss = nn.CrossEntropyLoss()

def encoder_optimal(x, sens):
    z = x.detach()
    z[:, 1] = torch.where(sens.bool(), z[:, 1] + 7, z[:, 1])
    return z

def train_adversary(args, d, p1, p2, encoder, train_loaders, valid_loaders):
    adv = nn.Sequential(
        nn.Linear(args.enc_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 2),
    )
    adv = adv.to(device)
    
    opt = optim.Adam(adv.parameters(), lr=1e-2)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.adv_epochs//3, gamma=0.1)
    
    for epoch in range(args.adv_epochs):
        for mode in ['train', 'valid']:
            if mode == 'train':
                data_loader1, data_loader2 = train_loaders
            else:
                data_loader1, data_loader2 = valid_loaders

            tot_adv_loss, tot_adv_acc, n_batches = 0, 0, 0
            for data1, data2 in zip(data_loader1, data_loader2):
                data_x1, targets1 = data1
                data_x2, targets2 = data2
                targets = torch.cat([targets1, targets2], dim=0)

                data_x1, data_x2 = data_x1.to(device), data_x2.to(device)
                targets = targets.to(device)
        
                x = torch.cat([data_x1, data_x2], dim=0)
                sens = torch.zeros(2*args.batch_size).long().to(device)
                sens[args.batch_size:] = 1

                z = encoder(x)
                # z = encoder_optimal(x, sens)

                # if args.plot:
                #     z1 = z[:args.batch_size].detach().cpu().numpy()
                #     z2 = z[args.batch_size:].detach().cpu().numpy()
                #     plt.scatter(z1[:, 0], z1[:, 1], color='blue')
                #     plt.scatter(z2[:, 0], z2[:, 1], color='red')
                #     plt.show()
                #     exit(0)

                if mode == 'train':
                    opt.zero_grad()

                pred = adv(z)
                y = pred.max(dim=1)[1]
                acc = (y == sens).float().mean()
                loss = ce_loss(pred, sens)
                if mode == 'train':
                    loss.backward()
                    opt.step()
                lr_scheduler.step()
                tot_adv_loss += loss.item()
                tot_adv_acc += acc.item()
                n_batches += 1
                # if epoch % 100 == 0:
                #     print('adv: ' , epoch, loss.item(), acc.item())
            if args.verbose and mode == 'train' and epoch % 10 == 0:
                print(epoch, mode, tot_adv_loss/n_batches, tot_adv_acc/n_batches)
    if args.verbose:
        print(f'final adv: loss={tot_adv_loss/n_batches:.4f}, acc={tot_adv_acc/n_batches:.4f}')
    else:
        print(f'{tot_adv_loss/n_batches:.4f} {tot_adv_acc/n_batches:.4f}')
    return adv


def train_encoder(args, d, train_loaders, valid_loaders):
    encoder = nn.Sequential(
        nn.Linear(2, 50),
        nn.ReLU(),
        nn.Linear(50, args.enc_dim),
    ).to(device)
    clf = nn.Sequential(
        nn.Linear(args.enc_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 2),
    ).to(device)
    adv = nn.Sequential(
        nn.Linear(args.enc_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 2),
    ).to(device)

    opt = optim.Adam(list(encoder.parameters()) + list(clf.parameters()), lr=1e-3)
    opt_adv = optim.Adam(adv.parameters(), lr=1e-3)

    prev_adv_acc = 0
    
    for epoch in range(args.n_epochs):
        for mode in ['train', 'valid']:
            if mode == 'train':
                data_loader1, data_loader2 = train_loaders
            else:
                data_loader1, data_loader2 = valid_loaders

            tot_clf_loss, tot_clf_acc, tot_adv_loss, tot_adv_acc, n_batches = 0, 0, 0, 0, 0
            for data1, data2 in zip(data_loader1, data_loader2):
                data_x1, targets1 = data1
                data_x2, targets2 = data2
                targets = torch.cat([targets1, targets2], dim=0)

                data_x1, data_x2 = data_x1.to(device), data_x2.to(device)
                targets = targets.to(device)

                if mode == 'train':
                    opt.zero_grad()
                    opt_adv.zero_grad()

                # data_x1 = p1.sample((args.batch_size,))
                # data_x2 = p2.sample((args.batch_size,))
                # targets = torch.cat([target_fn(data_x1), target_fn(data_x2)], dim=0)

                x = torch.cat([data_x1, data_x2], dim=0)
                sens = torch.zeros(2*args.batch_size).long().to(device)
                sens[args.batch_size:] = 1

                z = encoder(x)        

                if mode == 'train':
                    # Train adversary
                    pred_sens = adv(z.detach())
                    adv_loss = ce_loss(pred_sens, sens)
                    adv_loss.backward()
                    opt_adv.step()

                # Compute adv to train clf
                z = encoder(x)        
                pred_sens = adv(z)
                adv_acc = (pred_sens.max(dim=1)[1] == sens).float().mean()
                adv_loss = ce_loss(pred_sens, sens)

                clf_out = clf(z)
                pred_targets = clf_out.max(dim=1)[1]
                acc = (pred_targets == targets).float().mean()
                pred_loss = ce_loss(clf_out, targets)

                tot_clf_loss += pred_loss.item()
                tot_clf_acc += acc.item()
                tot_adv_loss += adv_loss.item()
                tot_adv_acc += adv_acc.item()
                n_batches += 1

                tot_loss = (1 - args.adv_coeff) * pred_loss - args.adv_coeff * adv_loss
                # tot_loss = -adv_loss
                if mode == 'train':
                    tot_loss.backward()
                    opt.step()
                    
            if args.verbose and (epoch+1) % 10 == 0:
                print('enc: [%s] epoch=%d, pred_loss=%.4f, clf_acc=%.4f, adv_loss=%.4f, adv_acc=%.4f' % (
                    mode, epoch+1,
                    tot_clf_loss/n_batches, tot_clf_acc/n_batches,
                    tot_adv_loss/n_batches, tot_adv_acc/n_batches))
            if mode == 'train':
                prev_adv_acc = tot_adv_acc/n_batches
    return encoder


def get_gaussians(y1, y2):
    mean1, cov1 = torch.FloatTensor([0, y1]), torch.FloatTensor([[1, 0], [0, 1]])
    mean2, cov2 = torch.FloatTensor([0, y2]), torch.FloatTensor([[1, 0], [0, 1]])
    p1 = MultivariateNormal(mean1, cov1)
    p2 = MultivariateNormal(mean2, cov2)
    target_fn = lambda x: (x[:, 0] > 0).long()
    return 2, p1, p2, target_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adv_coeff', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--adv_epochs', type=int, default=100)
    parser.add_argument('--enc_dim', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_train', type=int, default=1024)
    parser.add_argument('--n_valid', type=int, default=1024)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # d, p1, p2, target_fn = get_gaussians(3.5, -3.5)
    d, (p1, p2), _, train_loaders, valid_loaders = get_stripes(args, 4, False)

    if args.plot:
        colors = sns.color_palette()
        data_x1 = p1.sample((200, )).detach().cpu().numpy()
        data_x2 = p2.sample((200, )).detach().cpu().numpy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

        marker_size = 10
        
        label_1 = ((data_x1[:, 0] < 0) == (data_x1[:, 1] < 0))
        ax.scatter(data_x1[label_1, 0], data_x1[label_1, 1], s=marker_size, marker='o', color=colors[0])
        ax.scatter(data_x1[~label_1, 0], data_x1[~label_1, 1], s=marker_size, marker='*', color=colors[0])

        label_1 = ((data_x2[:, 0] < 0) == (data_x2[:, 1] < 0))
        ax.scatter(data_x2[label_1, 0], data_x2[label_1, 1], s=marker_size, marker='o', color=colors[1])
        ax.scatter(data_x2[~label_1, 0], data_x2[~label_1, 1], s=marker_size, marker='*', color=colors[1])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')

        fig.tight_layout()
        plt.savefig('gauss.pdf', bbox_inches='tight')
        plt.show()
        exit(0)

    encoder = train_encoder(args, d, train_loaders, valid_loaders)
    adv = train_adversary(args, d, p1, p2, encoder, train_loaders, valid_loaders)


    
if __name__ == '__main__':
    main()
