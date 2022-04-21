import argparse
import csv
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--with_test', action='store_true')
args = parser.parse_args()

seeds = [100, 101, 102, 103, 104]
batch_size = 256

prior_epochs = 80
adv_epochs = 80
n_epochs = 80
kl_start = 0
kl_end = 10
log_epochs = 10

lr = 1e-3
weight_decay = 0.0
n_blocks = 6

p_val = 0.2
p_test = 0.2

for seed in seeds:
    out_file = f'logs/health/health_{seed}.csv'
    
    with open(f'{out_file}', 'w') as csvfile:
        field_names = ['gamma', 'stat_dist', 'valid_unbal_acc', 'valid_bal_acc', 'test_unbal_acc', 'test_bal_acc', 'adv_valid_acc', 'adv_test_acc', 'test_dem_par', 'test_eq_0', 'test_eq_1']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

    for gamma in [0, 0.05, 0.1, 0.5, 0.95]:
        print(f'Running gamma={gamma}')
        cmd = f'python health_flow_multi.py --load --prior flow --prior_epochs {prior_epochs} --batch_size {batch_size} --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma {gamma} --seed {seed} --kl_start {kl_start} --kl_end {kl_end} --log_epochs {log_epochs} --lr {lr} --weight_decay {weight_decay} --n_blocks {n_blocks} --p_val {p_val} --p_test {p_test} --out_file {out_file}'
        if args.with_test:
            cmd += ' --with_test'
        print(cmd)
        os.system(cmd)



    

