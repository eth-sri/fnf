import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
from generative.real_nvp import FlowNetwork
from tqdm import tqdm



def train_flow_prior_main(y, args, q, train, train_loaders, valid, device):
    train1_loader, train2_loader = train_loaders
    train1, train2, targets1, targets2 = train
    valid1, valid2, v_targets1, v_targets2 = valid
    in_dim = train1.shape[1]

    mean, cov = torch.zeros(in_dim).to(device), torch.eye(in_dim).to(device)
    p_z = MultivariateNormal(mean, cov)
    prior1 = FlowNetwork(p_z, in_dim, [20], 4).to(device)
    prior2 = FlowNetwork(p_z, in_dim, [20], 4).to(device)
    opt_flow1 = optim.Adam(prior1.parameters(), lr=1e-2, weight_decay=1e-4)
    opt_flow2 = optim.Adam(prior2.parameters(), lr=1e-2, weight_decay=1e-4)
    lr_scheduler1 = optim.lr_scheduler.MultiStepLR(opt_flow1, milestones=[args.prior_epochs//3, 2*args.prior_epochs//3], gamma=0.1)
    lr_scheduler2 = optim.lr_scheduler.MultiStepLR(opt_flow2, milestones=[args.prior_epochs//3, 2*args.prior_epochs//3], gamma=0.1)

    v1 = torch.clamp(valid1 + q * torch.rand(valid1.shape).to(device), args.alpha/2, 1-args.alpha/2).logit().detach()
    v2 = torch.clamp(valid2 + q * torch.rand(valid2.shape).to(device), args.alpha/2, 1-args.alpha/2).logit().detach()

    if args.load_prior:
        prior1.load_state_dict(torch.load('logs/health/prior_flow1.pt', map_location=device))
        prior2.load_state_dict(torch.load('logs/health/prior_flow2.pt', map_location=device))
        with torch.no_grad():
            curr_valid_loss1 = -prior1.log_prob(v1).mean().item()
            curr_valid_loss2 = -prior2.log_prob(v2).mean().item()
        print('valid: ', curr_valid_loss1, curr_valid_loss2)
        return [prior1], [prior2]

    best_prior1_dict, best_valid_loss1 = None, None
    best_prior2_dict, best_valid_loss2 = None, None

    for epoch in range(args.prior_epochs):
        pbar = tqdm(zip(train1_loader, train2_loader))
        tot_loss1, tot_loss2, n_batches = 0, 0, 0
        with torch.no_grad():
            curr_valid_loss1 = -prior1.log_prob(v1).mean().item()
            curr_valid_loss2 = -prior2.log_prob(v2).mean().item()
        for (inputs1, targets1), (inputs2, targets2) in pbar:
            opt_flow1.zero_grad()
            opt_flow2.zero_grad()
            if y is not None:
                inputs1 = inputs1[targets1 == y]
                inputs2 = inputs2[targets2 == y]
                if inputs1.shape[0] == 0 or inputs2.shape[0] == 0:
                    continue
            if q is not None:
                t1 = torch.clamp(inputs1 + q * torch.rand(inputs1.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
                t2 = torch.clamp(inputs2 + q * torch.rand(inputs2.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
            else:
                t1 = inputs1
                t2 = inputs2
            _, logp1 = prior1.inverse(t1)
            _, logp2 = prior2.inverse(t2)
            loss1 = -logp1.mean()
            loss2 = -logp2.mean()
            loss = loss1 + loss2
            loss.backward()
            opt_flow1.step()
            opt_flow2.step()
            tot_loss1 += loss1.item()
            tot_loss2 += loss2.item()
            n_batches += 1
            pbar.set_description('prior: [train] epoch=%d, loss1=%.4lf, loss2=%.4lf, valid_loss1=%.4f, valid_loss2=%.4f' % (
                epoch, tot_loss1/n_batches, tot_loss2/n_batches, curr_valid_loss1, curr_valid_loss2
            ))

        lr_scheduler1.step()
        lr_scheduler2.step()

        if (not args.no_early_stop) and (epoch+1) % 10 == 0:
            if best_valid_loss1 is None or curr_valid_loss1 < best_valid_loss1:
                best_valid_loss1 = curr_valid_loss1
                best_prior1_dict = prior1.state_dict()
            if best_valid_loss2 is None or curr_valid_loss2 < best_valid_loss2:
                best_valid_loss2 = curr_valid_loss2
                best_prior2_dict = prior2.state_dict()
    print('best valid losses: ', best_valid_loss1, best_valid_loss2)
    if best_valid_loss1 is not None:
        prior1.load_state_dict(best_prior1_dict)
        prior2.load_state_dict(best_prior2_dict)

    for parameter in prior1.parameters():
        parameter.requires_grad_(False)
    for parameter in prior2.parameters():
        parameter.requires_grad_(False)

    return prior1, prior2


def train_flow_prior(args, q, train, train_loaders, valid, device):
    if args.fair_criterion == 'stat_parity':
        prior1, prior2 = train_flow_prior_main(None, args, q, train, train_loaders, valid, device)
        prior1, prior2 = [prior1], [prior2]
    elif args.fair_criterion == 'eq_opp':
        prior1, prior2 = train_flow_prior_main(1, args, q, train, train_loaders, valid, device)
        prior1, prior2 = [prior1], [prior2]
    else:
        prior1_0, prior2_0 = train_flow_prior_main(0, args, q, train, train_loaders, valid, device)
        prior1_1, prior2_1 = train_flow_prior_main(1, args, q, train, train_loaders, valid, device)
        prior1, prior2 = [prior1_0, prior1_1], [prior2_0, prior2_1]
    return prior1, prior2
