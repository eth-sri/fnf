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
weight_decay = 1e-3
n_blocks = 4

if args.with_test:
    p_val = 0.01
    p_test = 0.2
else:
    p_val = 0.2
    p_test = 0.2

for seed in seeds:
    out_file = f'logs/compas/compas_{seed}.csv'

    with open(f'{out_file}', 'w') as csvfile:
        field_names = ['gamma', 'stat_dist', 'valid_unbal_acc', 'valid_bal_acc', 'test_unbal_acc', 'test_bal_acc', 'adv_valid_acc', 'adv_test_acc']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

    for gamma in [0, 0.02, 0.1, 0.2, 0.9]:
        print(f'Running gamma={gamma}')
        cmd = f'python train_enc_categorical.py --dataset compas --seed {seed} --batch_size 128 --weight_decay {weight_decay} --n_epochs 100 --adv_epochs 100 --gamma {gamma} --p_val {p_val} --p_test {p_test} --out_file {out_file} --verbose --encode'
        if args.with_test:
            cmd += ' --with_test'
        print(cmd)
        os.system(cmd)


    

