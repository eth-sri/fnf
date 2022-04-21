import argparse
import csv
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--with_test', action='store_true')
parser.add_argument('--gamma', type=float)
args = parser.parse_args()

if args.gamma is None:
    gammas = [0, 0.2]
else:
    gammas = [args.gamma]

seeds = [100] #, 101, 102, 103, 104]
batch_size = 256

prior_epochs = 60
adv_epochs = 60
n_epochs = 60
kl_start = 0
kl_end = 5
log_epochs = 10

lr = 1e-3
weight_decay = 0.0
n_blocks = 6

prior = 'gmm'
gmm_comps1 = 10
gmm_comps2 = 10

labels = ['MSC2a3', 'METAB3']
# labels = ['MSC2a3', 'METAB3', 'ARTHSPIN', 'NEUMENT', 'RESPR4']

if args.with_test:
    p_val = 0.2
    p_test = 0.2
else:
    p_val = 0.2
    p_test = 0.2

for seed in seeds:
    for label in ['default'] + labels:
        if args.gamma is None:
            out_file = f'logs/health/transfer/health_{seed}_{label}.csv'
        else:
            out_file = f'logs/health/transfer/health_{args.gamma}_{seed}_{label}.csv'

        with open(f'{out_file}', 'w') as csvfile:
            field_names = ['gamma', 'stat_dist', 'valid_unbal_acc', 'valid_bal_acc', 'test_unbal_acc', 'test_bal_acc', 'adv_valid_acc', 'adv_test_acc']
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
        
    for gamma in gammas:
        print(f'Running gamma={gamma}')

        if args.gamma is None:
            default_out_file = f'logs/health/transfer/health_{seed}_default.csv'
        else:
            default_out_file = f'logs/health/transfer/health_{args.gamma}_{seed}_default.csv'
        
        default_cmd = f'python health_flow_multi.py --load --prior flow --prior_epochs {prior_epochs} --batch_size {batch_size} --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma {gamma} --seed {seed} --kl_start {kl_start} --kl_end {kl_end} --log_epochs {log_epochs} --lr {lr} --weight_decay {weight_decay} --n_blocks {n_blocks} --p_val {p_val} --p_test {p_test} --out_file {default_out_file} --transfer --save_enc'
        if args.with_test:
            default_cmd += ' --with_test'
        os.system(default_cmd)

        for label in labels:
            if args.gamma is None:
                out_file = f'logs/health/transfer/health_{seed}_{label}.csv'
            else:
                out_file = f'logs/health/transfer/health_{args.gamma}_{seed}_{label}.csv'
            cmd = f'python health_flow_multi.py --load --prior flow --prior_epochs {prior_epochs} --batch_size {batch_size} --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma {gamma} --seed {seed} --kl_start {kl_start} --kl_end {kl_end} --log_epochs {log_epochs} --lr {lr} --weight_decay {weight_decay} --n_blocks {n_blocks} --p_val {p_val} --p_test {p_test} --out_file {out_file} --transfer --load_enc --label PrimaryConditionGroup={label}'
            print(cmd)
            if args.with_test:
                cmd += ' --with_test'
            os.system(cmd)




    

