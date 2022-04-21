import argparse
import csv
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from real_nvp_encoder import FlowEncoder
from neural_spline_encoder import NeuralSplineEncoder
from torch.distributions import MultivariateNormal, Categorical
from tqdm import tqdm

device = 'cuda'


def get_log_pz(z, prior, flows):
    x, sum_logp = flows[0].forward(z)
    prior_logp = prior.log_prob(x)
    return sum_logp + prior_logp


def predict(inputs, targets, flows, clf):
    z = flows[0].inverse(inputs)
    out = clf(z)
    y = out.max(dim=1)[1]
    acc, pred_loss = (y == targets).float(), F.cross_entropy(out, targets, reduction='none')
    return acc, pred_loss


def predict_avg(inputs, targets, flows, clf):
    acc, pred_loss = 0, 0
    z = flows[0].inverse(inputs)[0]
    out = clf(z)
    y = out.max(dim=1)[1]
    acc = (y == targets).float()
    pos = (y == 1).float()
    pos_0 = ((y == 1) & (targets == 0)).float()
    pos_1 = ((y == 1) & (targets == 1)).float()
    pred_loss = F.cross_entropy(out, targets, reduction='none')
    return (pos, pos_0, pos_1), acc, pred_loss


def evaluate(args, q, data1_loader, data2_loader, flows, clf):
    tot_data_acc1, tot_data_pos1, tot_data_pos1_0, tot_data_pos1_1, tot_data_loss1 = 0, 0, 0, 0, 0
    tot_samples1, tot_samples1_0, tot_samples1_1 = 0, 0, 0
    for inputs, targets in data1_loader:
        inputs_tf = inputs if q is None else torch.clamp(inputs + q * torch.rand(inputs.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
        (data_pos1, data_pos1_0, data_pos1_1), data_acc1, data_loss1 = predict_avg(inputs_tf, targets, flows[0], clf)
        tot_data_pos1 += data_pos1.sum().item()
        tot_data_pos1_0 += data_pos1_0.sum().item()
        tot_data_pos1_1 += data_pos1_1.sum().item()
        tot_data_acc1 += data_acc1.sum().item()
        tot_data_loss1 += data_loss1.sum().item()
        tot_samples1 += targets.shape[0]
        tot_samples1_0 += (targets == 0).float().sum().item()
        tot_samples1_1 += (targets == 1).float().sum().item()
    # print(tot_data_pos1/tot_samples1, tot_data_pos1_0/tot_samples1_0, tot_data_pos1_1/tot_samples1_1)
    # exit(0)
    tot_data_acc2, tot_data_pos2, tot_data_pos2_0, tot_data_pos2_1, tot_data_loss2 = 0, 0, 0, 0, 0
    tot_samples2, tot_samples2_0, tot_samples2_1 = 0, 0, 0
    for inputs, targets in data2_loader:
        inputs_tf = inputs if q is None else torch.clamp(inputs + q * torch.rand(inputs.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
        (data_pos2, data_pos2_0, data_pos2_1), data_acc2, data_loss2 = predict_avg(inputs_tf, targets, flows[1], clf)
        tot_data_pos2 += data_pos2.sum().item()
        tot_data_pos2_0 += data_pos2_0.sum().item()
        tot_data_pos2_1 += data_pos2_1.sum().item()
        tot_data_acc2 += data_acc2.sum().item()
        tot_data_loss2 += data_loss2.sum().item()
        tot_samples2 += targets.shape[0]
        tot_samples2_0 += (targets == 0).float().sum().item()
        tot_samples2_1 += (targets == 1).float().sum().item()

    dem_par = abs(tot_data_pos1/tot_samples1 - tot_data_pos2/tot_samples2)
    eq_0 = abs(tot_data_pos1_0/tot_samples1_0 - tot_data_pos2_0/tot_samples2_0)
    eq_1 = abs(tot_data_pos1_1/tot_samples1_1 - tot_data_pos2_1/tot_samples2_1)
    
    data_unbal_acc = ((tot_data_acc1 + tot_data_acc2)/(tot_samples1+tot_samples2))
    data_bal_acc = 0.5 * (tot_data_acc1/tot_samples1 + tot_data_acc2/tot_samples2)
    data_loss = 0.5 * (tot_data_loss1/tot_samples1 + tot_data_loss2/tot_samples2)
    return (dem_par, eq_0, eq_1), data_unbal_acc, data_bal_acc, data_loss

def evaluate_adversary(y, args, q, adv, data1_loader, data2_loader, flows):
    tot_data_acc1, tot_data_loss1, tot_samples1 = 0, 0, 0
    tot_data_acc2, tot_data_loss2, tot_samples2 = 0, 0, 0
    for inputs, targets in data1_loader:
        if y is not None:
            inputs = inputs[targets == y]
            targets = targets[targets == y]
        if inputs.shape[0] == 0:
            continue
        inputs_tf = inputs if q is None else torch.clamp(inputs + q * torch.rand(inputs.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
        sens = torch.zeros(inputs.shape[0]).long().to(device)
        z = flows[0][0].inverse(inputs_tf)[0]
        pred = adv(z)
        data_acc1 = (pred.max(dim=1)[1] == sens).float()
        data_loss1 = F.cross_entropy(pred, sens, reduction='none')
        tot_data_acc1 += data_acc1.sum()
        tot_data_loss1 += data_loss1.sum()
        tot_samples1 += inputs.shape[0]
    for inputs, targets in data2_loader:
        if y is not None:
            inputs = inputs[targets == y]
            targets = targets[targets == y]
        if inputs.shape[0] == 0:
            continue
        inputs_tf = inputs if q is None else torch.clamp(inputs + q * torch.rand(inputs.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
        sens = torch.ones(inputs.shape[0]).long().to(device)
        z = flows[1][0].inverse(inputs_tf)[0]
        pred = adv(z)
        data_acc2 = (pred.max(dim=1)[1] == sens).float()
        data_loss2 = F.cross_entropy(pred, sens, reduction='none')
        tot_data_acc2 += data_acc2.sum()
        tot_data_loss2 += data_loss2.sum()
        tot_samples2 += inputs.shape[0]
    data_acc = 0.5 * (tot_data_acc1/tot_samples1 + tot_data_acc2/tot_samples2).item()
    data_loss = 0.5 * (tot_data_loss1/tot_samples1 + tot_data_loss2/tot_samples2).item()
    return data_acc, data_loss


def get_total(data_loader):
    ret = 0
    for inputs, _ in data_loader:
        ret += inputs.shape[0]
    return ret


def train_adversary(y, args, d, q, flows, train_loaders, valid_loaders, test_loaders, no_encode=False):
    train1_loader, train2_loader = train_loaders
    valid1_loader, valid2_loader = valid_loaders
    if test_loaders is not None:
        test1_loader, test2_loader = test_loaders

    adv = nn.Sequential(
        nn.Linear(d, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 2),
    )
    adv = adv.to(device)
    batch_size = args.batch_size

    opt = optim.Adam(adv.parameters(), lr=1e-2)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[args.adv_epochs//3, 2*args.adv_epochs//3], gamma=0.1)

    for epoch in range(args.adv_epochs):
        tot_loss1, tot_loss2, tot_acc1, tot_acc2, tot_samples1, tot_samples2 = 0, 0, 0, 0, 0, 0
        for train1, train2 in zip(train1_loader, train2_loader):
            inputs1, targets1 = train1
            inputs2, targets2 = train2
            opt.zero_grad()

            if y is not None:
                inputs1 = inputs1[targets1 == y]
                inputs2 = inputs2[targets2 == y]
                targets1 = targets1[targets1 == y]
                targets2 = targets2[targets2 == y]

                if inputs1.shape[0] == 0 or inputs2.shape[0] == 0:
                    continue
            
            if q is not None:
                data_x1 = torch.clamp(inputs1 + q * torch.rand(inputs1.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
                data_x2 = torch.clamp(inputs2 + q * torch.rand(inputs2.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
            else:
                data_x1 = inputs1
                data_x2 = inputs2

            sens_x1 = torch.zeros(inputs1.shape[0]).long().to(device)
            sens_x2 = torch.ones(inputs2.shape[0]).long().to(device)

            if no_encode:
                # Sanity check: don't use our encoding
                x1_z1, x2_z2 = data_x1, data_x2
            else:
                x1_z1 = flows[0][0].inverse(data_x1)[0]
                x2_z2 = flows[1][0].inverse(data_x2)[0]

            x1_pred, x2_pred = adv(x1_z1), adv(x2_z2)
            x1_y, x2_y = x1_pred.max(dim=1)[1], x2_pred.max(dim=1)[1]
            acc1, acc2 = (x1_y == sens_x1).float(), (x2_y == sens_x2).float()
            loss1, loss2 = F.cross_entropy(x1_pred, sens_x1), F.cross_entropy(x2_pred, sens_x2)

            loss = 0.5 * (loss1.mean() + loss2.mean())
            loss.backward()
            opt.step()

            tot_loss1 += loss1.sum().item()
            tot_loss2 += loss2.sum().item()
            tot_acc1 += acc1.sum().item()
            tot_acc2 += acc2.sum().item()
            tot_samples1 += inputs1.shape[0]
            tot_samples2 += inputs2.shape[0]

        if (epoch+1) % args.log_epochs == 0:
            print('[train adv] epoch=%d, acc=%.4f, loss=%.4f' % (
                epoch+1,
                0.5 * (tot_acc1/tot_samples1 + tot_acc2/tot_samples2),
                0.5 * (tot_loss1/tot_samples1 + tot_loss2/tot_samples2),
            ))
        lr_scheduler.step()

    with torch.no_grad():
        adv_valid_acc, adv_valid_loss = evaluate_adversary(
            y, args, q, adv, valid1_loader, valid2_loader, flows)
        if test_loaders is not None:
            adv_test_acc, adv_test_loss = evaluate_adversary(
                y, args, q, adv, test1_loader, test2_loader, flows)
        else:
            adv_test_acc, adv_test_loss = -1, -1
            
    return adv_valid_acc, adv_valid_loss, adv_test_acc, adv_test_loss


def train_flow(args, d, q, priors_1, priors_2, flow_dims, clf_dims, train_loaders, valid_loaders, test_loaders, is_spline=False, return_loss_bounds=False, lb_pred_loss=0, ub_pred_loss=1, lb_kl_loss=0, ub_kl_loss=1):
    train1_loader, train2_loader = train_loaders
    valid1_loader, valid2_loader = valid_loaders
    if test_loaders is not None:
        test1_loader, test2_loader = test_loaders

    # Estimate TV
    tv_nets = [nn.Sequential(
        nn.Linear(d, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 1),
        nn.Tanh(),
    ).to(device) for _ in range(2)]

    # opt = optim.Adam(list(tv_nets[0].parameters()) + list(tv_nets[1].parameters()), lr=1e-2)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[args.n_epochs//3, 2*args.n_epochs//3], gamma=0.1)
    # for epoch in range(args.n_epochs):
    #     for mode in ['train', 'valid']:
    #         if mode == 'train':
    #             data_loaders = zip(train1_loader, train2_loader)
    #         else:
    #             data_loaders = zip(valid1_loader, valid2_loader) if not args.with_test else zip(test1_loader, test2_loader)
    #         tot_tv1, tot_tv2, n_batches = 0, 0, 0
    #         for data1, data2 in data_loaders:
    #             opt.zero_grad()
    #             inputs1, inputs2 = data1[0], data2[0]
    #             if q is not None:
    #                 inputs1_tf = torch.clamp(inputs1 + q * torch.rand(inputs1.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
    #                 inputs2_tf = torch.clamp(inputs2 + q * torch.rand(inputs2.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
    #             prior_inputs1 = priors_1[0].sample((args.batch_size,)).to(device)
    #             prior_inputs2 = priors_2[0].sample((args.batch_size,)).to(device)
    #             prior_outs1 = tv_nets[0](prior_inputs1)
    #             prior_outs2 = tv_nets[1](prior_inputs2)
    #             real_outs1 = tv_nets[0](inputs1_tf)
    #             real_outs2 = tv_nets[1](inputs2_tf)
    #             tv1 = 0.5 * (prior_outs1.mean(0) - real_outs1.mean(0))
    #             tv2 = 0.5 * (prior_outs2.mean(0) - real_outs2.mean(0))
    #             if mode == 'train':
    #                 (-(tv1 + tv2)).backward()
    #                 opt.step()
    #             tot_tv1 += tv1.item()
    #             tot_tv2 += tv2.item()
    #             n_batches += 1
    #         if epoch % 10 == 0:
    #             print(mode, epoch, tot_tv1/n_batches, tot_tv2/n_batches)
    #     lr_scheduler.step()
    # exit(0)

    n_flows = args.n_flows

    tot1, tot2 = get_total(train1_loader), get_total(train2_loader)

    masks = []
    for i in range(20):
        t = np.array([j % 2 for j in range(d)])
        np.random.shuffle(t)
        masks += [t, 1 - t]

    if is_spline:
        flows = [[NeuralSplineEncoder(None, d, flow_dims, args.n_blocks, masks).to(device) for i in range(n_flows)]
                 for _ in range(2)]
    else:
        flows = [[FlowEncoder(None, d, flow_dims, args.n_blocks, masks).to(device) for i in range(n_flows)]
                 for _ in range(2)]
    batch_size = args.batch_size

    for i in range(n_flows):
        flows[0][i].load_state_dict(flows[1][i].state_dict())
        
    if len(clf_dims) > 0:
        clf_layers = [nn.Linear(d, clf_dims[0]), nn.ReLU()]
        for i in range(1, len(clf_dims)):
            clf_layers += [nn.Linear(clf_dims[i-1], clf_dims[i]), nn.ReLU()]
        clf_layers += [nn.Linear(clf_dims[-1], 2)]
    else:
        clf_layers = [nn.Linear(d, 2)]
    clf = nn.Sequential(*clf_layers).to(device)

    flow_params = []
    for f in flows[0]:
        flow_params += list(f.parameters())
    for f in flows[1]:
        flow_params += list(f.parameters())

    if args.load_enc:
        print('Loading encoders and freezing them')
        flows[0][0].load_state_dict(torch.load('trained_models/flow_0_0_enc.pt', map_location=device))
        flows[1][0].load_state_dict(torch.load('trained_models/flow_1_0_enc.pt', map_location=device))
        flow_params = []
        for param in flows[0][0].parameters():
            param.requires_grad_(False)
        for param in flows[1][0].parameters():
            param.requires_grad_(False)

    opt = optim.Adam(list(clf.parameters()) + flow_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[args.n_epochs//3, 2*args.n_epochs//3], gamma=0.1)

    D = 100
    W = torch.randn((d, D)).to(device)
    b = torch.rand((1, D)).to(device)

    min_kl_loss = min_pred_loss = float('inf')
    max_kl_loss = max_pred_loss = float('-inf')

    for epoch in range(args.n_epochs):
        tot_kl1, tot_kl2, tot_e1, tot_e2, tot_pred1, tot_pred2, tot_acc, tot_acc1, tot_acc2 = 0, 0, 0, 0, 0, 0, 0, 0, 0
        tot_samples1, tot_samples2 = 0, 0
        pbar = tqdm(zip(train1_loader, train2_loader))
        for train1, train2 in pbar:
            inputs1, targets1 = train1
            inputs2, targets2 = train2
            
            kl_loss, e1, e2 = 0, 0, 0
            for p1, p2 in zip(priors_1, priors_2):
                data_x1 = p1.sample((inputs1.shape[0],)).to(device)
                data_x2 = p2.sample((inputs2.shape[0],)).to(device)
                # data_x1 = p1.sample((batch_size,)).to(device)
                # data_x2 = p2.sample((batch_size,)).to(device)
                opt.zero_grad()

                # first
                x1_z1, x1_logp1 = flows[0][0].inverse(data_x1)
                x1_x2, x1_logp2 = flows[1][0].forward(x1_z1)

                logprob_1 = get_log_pz(x1_z1, p1, flows[0])
                logprob_2 = get_log_pz(x1_z1, p2, flows[1])
                x1_logp1 = x1_logp1 + p1.log_prob(data_x1)
                x1_logp2 = x1_logp2 + p2.log_prob(x1_x2)

                # c1 = x1_z1.mean(0)
                kl1 = (x1_logp1 - x1_logp2)
                e1 += (x1_logp1 > x1_logp2).float()

                # second
                x2_z2, x2_logp2 = flows[1][0].inverse(data_x2)
                x2_x1, x2_logp1 = flows[0][0].forward(x2_z2)

                x2_logp2 = x2_logp2 + p2.log_prob(data_x2)
                x2_logp1 = x2_logp1 + p1.log_prob(x2_x1)

                # c2 = x2_z2.mean(0)
                kl2 = (x2_logp2 - x2_logp1)
                e2 += (x2_logp1 > x2_logp2).float()

                # mean_loss = ((c1 - c2)**2).sum()
                kl_loss = kl_loss + 0.5 * (kl1.mean() + kl2.mean())
            if len(priors_1) > 1:
                e1, e2, kl_loss = 0.5 * e1, 0.5 * e2, 0.5 * kl_loss

            if q is not None:
                inputs1_tf = torch.clamp(inputs1 + q * torch.rand(inputs1.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
                inputs2_tf = torch.clamp(inputs2 + q * torch.rand(inputs2.shape).to(device), args.alpha/2, 1-args.alpha/2).logit()
            else:
                inputs1_tf = inputs1
                inputs2_tf = inputs2

            _, acc1, pred_loss1 = predict_avg(inputs1_tf, targets1, flows[0], clf)
            _, acc2, pred_loss2 = predict_avg(inputs2_tf, targets2, flows[1], clf)

            if epoch < args.kl_start:
                curr_gamma = 0.0
            else:
                curr_gamma = args.gamma * min((epoch+1-args.kl_start)/float(args.kl_end-args.kl_start), 1.0)
            # prob_loss = mix * kl_loss + (1 - mix) * mean_loss
            pred_loss = 0.5 * (pred_loss1.mean() + pred_loss2.mean())

            if args.scalarization == 'convex' or epoch < args.kl_end:
                loss = curr_gamma * kl_loss + (1 - curr_gamma) * pred_loss
            elif args.scalarization == 'chebyshev':
                loss = max(
                    curr_gamma * (kl_loss - lb_kl_loss) / (ub_kl_loss - lb_kl_loss),
                    (1 - curr_gamma) * (pred_loss - lb_pred_loss) / (ub_pred_loss - lb_pred_loss)
                )

            if args.kl_end <= epoch + 1:
                min_pred_loss = min(min_pred_loss, pred_loss.item())
                max_pred_loss = max(max_pred_loss, pred_loss.item())
                min_kl_loss = min(min_kl_loss, kl_loss.item())
                max_kl_loss = max(max_kl_loss, kl_loss.item())

            loss.backward()
            opt.step()

            tot_kl1 += kl1.sum().item()
            tot_kl2 += kl2.sum().item()
            tot_e1 += e1.sum().item()
            tot_e2 += e2.sum().item()
            tot_acc1 += acc1.sum().item()
            tot_acc2 += acc2.sum().item()
            tot_pred1 += pred_loss1.sum().item()
            tot_pred2 += pred_loss2.sum().item()
            tot_samples1 += inputs1.shape[0]
            tot_samples2 += inputs2.shape[0]

            pbar.set_description('epoch: %d, curr_gamma: %.4f, kl1: %.3f, kl2: %.3f, e1: %.3lf, e2: %.3lf, pred_loss: %.3f, unbal_acc: %.3f, bal_acc: %.3f' % (
                epoch, curr_gamma, tot_kl1/tot_samples1, tot_kl2/tot_samples1, tot_e1/tot_samples1, tot_e2/tot_samples2,
                0.5*(tot_pred1/tot_samples1 + tot_pred2/tot_samples2),
                (tot_acc1+tot_acc2)/(tot_samples1+tot_samples2),
                0.5*(tot_acc1/tot_samples1 + tot_acc2/tot_samples2)))

        lr_scheduler.step()
        if (epoch+1) % args.log_epochs == 0:
            with torch.no_grad():
                (valid_dem_par, valid_eq_0, valid_eq_1), valid_unbal_acc, valid_bal_acc, valid_loss = evaluate(args, q, valid1_loader, valid2_loader, flows, clf)
                print(f'[valid] dem_par={valid_dem_par:.4f}, eq_0={valid_eq_0:.4f}, eq_1={valid_eq_1:.4f}, unbal_acc={valid_unbal_acc:.4f}, bal_acc={valid_bal_acc:.4f}, loss={valid_loss:.4f}')
                if test_loaders is not None:
                    (test_dem_par, test_eq_0, test_eq_1), test_unbal_acc, test_bal_acc, test_loss = evaluate(args, q, test1_loader, test2_loader, flows, clf)
                    print(f'[test] dem_par={test_dem_par:.4f}, eq_0={test_eq_0:.4f}, eq_1={test_eq_1:.4f}, unbal_acc={test_unbal_acc}, bal_acc={test_bal_acc:.4f}, loss={test_loss:.4f}')

    if args.save_enc:
        torch.save(flows[0][0].state_dict(), 'trained_models/flow_0_0_enc.pt')
        torch.save(flows[1][0].state_dict(), 'trained_models/flow_1_0_enc.pt')
        torch.save(clf.state_dict(), 'trained_models/clf.pt')

    print('')
    dist_samples = 2000
    with torch.no_grad():
        dists_y = []
        for p1, p2 in zip(priors_1, priors_2):
            data_x1 = p1.sample((dist_samples,)).to(device)
            data_x2 = p2.sample((dist_samples,)).to(device)

            x1_z1 = flows[0][0].inverse(data_x1)[0]
            logprob_1 = get_log_pz(x1_z1, p1, flows[0])
            logprob_2 = get_log_pz(x1_z1, p2, flows[1])
            e1 = (logprob_1 > logprob_2).float().mean()

            x2_z2 = flows[1][0].inverse(data_x2)[0]
            logprob_1 = get_log_pz(x2_z2, p1, flows[0])
            logprob_2 = get_log_pz(x2_z2, p2, flows[1])
            e2 = (logprob_1 > logprob_2).float().mean()

            dists_y += [torch.abs(e1-e2).item()]
        dist = np.sum(dists_y)
            
        with torch.no_grad():
            with torch.no_grad():
                (valid_dem_par, valid_eq_0, valid_eq_1), valid_unbal_acc, valid_bal_acc, valid_loss = evaluate(args, q, valid1_loader, valid2_loader, flows, clf)
                print(f'[valid] dem_par={valid_dem_par:.4f}, eq_0={valid_eq_0:.4f}, eq_1={valid_eq_1:.4f}, unbal_acc={valid_unbal_acc:.4f}, bal_acc={valid_bal_acc:.4f}, loss={valid_loss:.4f}')
                if test_loaders is not None:
                    (test_dem_par, test_eq_0, test_eq_1), test_unbal_acc, test_bal_acc, test_loss = evaluate(args, q, test1_loader, test2_loader, flows, clf)
                    print(f'[test] dem_par={test_dem_par:.4f}, eq_0={test_eq_0:.4f}, eq_1={test_eq_1:.4f}, unbal_acc={test_unbal_acc}, bal_acc={test_bal_acc:.4f}, loss={test_loss:.4f}')
                else:
                    test_dem_par, test_eq_0, test_eq_1, test_unbal_acc, test_bal_acc, test_loss = -1, -1, -1, -1, -1, -1

        print('stat distance: ', dists_y)
        print(f'[clf final valid] dem_par={valid_dem_par:.4f}, eq_0={valid_eq_0:.4f}, eq_1={valid_eq_1:.4f}, unbal_acc={valid_unbal_acc:.4f}, bal_acc={valid_bal_acc:.4f}, loss={valid_loss:.4f}')
        print(f'[clf final test] dem_par={test_dem_par:.4f}, eq_0={test_eq_0:.4f}, eq_1={test_eq_1:.4f}, unbal_acc={test_unbal_acc:.4f}, bal_acc={test_bal_acc:.4f}, loss={test_loss:.4f}')

    # Training adversary
    if args.fair_criterion == 'eq_odds':
        adv_valid_acc, adv_valid_loss, adv_test_acc, adv_test_loss = 0, 0, 0, 0
        for y in range(2):
            adv_valid_acc_y, adv_valid_loss_y, adv_test_acc_y, adv_test_loss_y = train_adversary(
                y, args, d, q, flows, train_loaders, valid_loaders, test_loaders)
            print(f'[adv final valid] y={y}, acc={adv_valid_acc_y:.4f}, loss={adv_valid_loss_y:.4f}')
            print(f'[adv final test] y={y}, acc={adv_test_acc_y:.4f}, loss={adv_test_loss_y:.4f}')
            adv_valid_acc += adv_valid_acc_y
            adv_valid_loss += adv_valid_loss_y
            adv_test_acc += adv_test_acc_y
            adv_test_loss += adv_test_loss_y
    elif args.fair_criterion == 'eq_opp':
        adv_valid_acc, adv_valid_loss, adv_test_acc, adv_test_loss = train_adversary(1, args, d, q, flows, train_loaders, valid_loaders, test_loaders)
        print(f'[adv final valid] y=1, acc={adv_valid_acc:.4f}, loss={adv_valid_loss:.4f}')
        print(f'[adv final test] y=1, acc={adv_test_acc:.4f}, loss={adv_test_loss:.4f}')
    else:
        adv_valid_acc, adv_valid_loss, adv_test_acc, adv_test_loss = train_adversary(None, args, d, q, flows, train_loaders, valid_loaders, test_loaders)
        print(f'[adv final valid] acc={adv_valid_acc:.4f}, loss={adv_valid_loss:.4f}')
        print(f'[adv final test] acc={adv_test_acc:.4f}, loss={adv_test_loss:.4f}')

    if return_loss_bounds:
        return flows, (min_pred_loss, max_pred_loss), (min_kl_loss, max_kl_loss)

    if args.out_file is not None:
        with open(args.out_file, 'a') as csvfile:
            field_names = ['gamma', 'stat_dist', 'valid_unbal_acc', 'valid_bal_acc', 'test_unbal_acc', 'test_bal_acc', 'adv_valid_acc', 'adv_test_acc', 'test_dem_par', 'test_eq_0', 'test_eq_1']
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerow({'gamma': args.gamma, 'stat_dist': dist,
                             'valid_unbal_acc': valid_unbal_acc, 'valid_bal_acc': valid_bal_acc,
                             'test_unbal_acc': test_unbal_acc, 'test_bal_acc': test_bal_acc,
                             'adv_valid_acc': adv_valid_acc, 'adv_test_acc': adv_test_acc,
                             'test_dem_par': test_dem_par, 'test_eq_0': test_eq_0, 'test_eq_1': test_eq_1,
            })

    return flows

