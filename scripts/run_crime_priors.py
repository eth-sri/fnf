import argparse
import csv
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--with_test', action='store_true')
args = parser.parse_args()

seeds = [100, 101, 102, 103, 104]
batch_size = 128

flow_prior_epochs = 100
autoreg_prior_epochs = 50
adv_epochs = 60
n_epochs = 60

kl_start = 0
kl_end = 50
log_epochs = 10

lr = 1e-2
weight_decay = 1e-4
n_blocks = 4

gmm_comps1 = 4
gmm_comps2 = 2


if args.with_test:
    p_val = 0.01
    p_test = 0.2
else:
    p_val = 0.2
    p_test = 0.2

for seed in seeds:
    for prior in ['autoreg', 'gmm', 'flow']:
        out_file = f'logs/crime/{prior}/crime_{seed}.csv'

        with open(f'{out_file}', 'w') as csvfile:
            field_names = ['gamma', 'stat_dist', 'valid_unbal_acc', 'valid_bal_acc', 'test_unbal_acc', 'test_bal_acc', 'adv_valid_acc', 'adv_test_acc']
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

        for gamma in [0, 0.02, 0.1, 0.2, 0.9]:
            print(f'Running gamma={gamma}')
            if prior == 'gmm':
                cmd = f'python crime_flow_multi.py --prior gmm --batch_size {batch_size} --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma {gamma} --seed {seed} --train_dec --kl_start {kl_start} --kl_end {kl_end} --log_epochs {log_epochs} --gmm_comps1 {gmm_comps1} --gmm_comps2 {gmm_comps2} --lr {lr} --weight_decay {weight_decay} --n_blocks {n_blocks} --p_val {p_val} --p_test {p_test} --out_file {out_file}'
            elif prior == 'flow':
                cmd = f'python crime_flow_multi.py --prior flow --prior_epochs {flow_prior_epochs} --batch_size {batch_size} --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma {gamma} --seed {seed} --train_dec --kl_start {kl_start} --kl_end {kl_end} --log_epochs {log_epochs} --lr {lr} --weight_decay {weight_decay} --n_blocks {n_blocks} --p_val {p_val} --p_test {p_test} --out_file {out_file}'
            elif prior == 'autoreg':
                cmd = f'python crime_flow_multi.py --prior flow --prior_epochs {autoreg_prior_epochs} --batch_size {batch_size} --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma {gamma} --seed {seed} --train_dec --kl_start {kl_start} --kl_end {kl_end} --log_epochs {log_epochs} --lr {lr} --weight_decay {weight_decay} --n_blocks {n_blocks} --p_val {p_val} --p_test {p_test} --out_file {out_file}'
            else:
                assert False
                
            if args.with_test:
                cmd += ' --with_test'
            print(cmd)
            os.system(cmd)


    

