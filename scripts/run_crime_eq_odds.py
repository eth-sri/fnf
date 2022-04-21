import argparse
import csv
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--with_test', action='store_true')
args = parser.parse_args()

seeds = [100, 101, 102, 103, 104]
batch_size = 128

adv_epochs = 60
n_epochs = 60
kl_start = 0
kl_end = 50
log_epochs = 10

lr = 1e-2
weight_decay = 1e-4
n_blocks = 4

prior = 'gmm'
gmm_comps1 = 4
gmm_comps2 = 2

if args.with_test:
    p_val = 0.01
    p_test = 0.2
else:
    p_val = 0.2
    p_test = 0.2

for seed in seeds:
    out_file = f'logs/crime/eq_odds/crime_{seed}.csv'

    with open(f'{out_file}', 'w') as csvfile:
        field_names = ['gamma', 'stat_dist', 'valid_unbal_acc', 'valid_bal_acc', 'test_unbal_acc', 'test_bal_acc', 'adv_valid_acc', 'adv_test_acc', 'test_dem_par', 'test_eq_0', 'test_eq_1']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

    # for gamma in [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.95]:
    for gamma in [0, 0.02, 0.1, 0.2, 0.9]:
        print(f'Running gamma={gamma}')
        cmd = f'python crime_flow_multi.py --prior gmm --batch_size {batch_size} --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma {gamma} --seed {seed} --train_dec --kl_start {kl_start} --kl_end {kl_end} --log_epochs {log_epochs} --gmm_comps1 {gmm_comps1} --gmm_comps2 {gmm_comps2} --lr {lr} --weight_decay {weight_decay} --n_blocks {n_blocks} --p_val {p_val} --p_test {p_test} --out_file {out_file} --fair_criterion eq_odds'
        if args.with_test:
            cmd += ' --with_test'
        print(cmd)
        os.system(cmd)


    

