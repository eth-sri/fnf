import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--with-test', action='store_true')
args = parser.parse_args()

seeds = [100, 101, 102, 103, 104]

parameters = {
    'adult': {
        'batch_size': 128, 'epochs': 100, 'weight_decay': 1e-3, 'dims': [20]
    },
    'compas': {
        'batch_size': 128, 'epochs': 100, 'weight_decay': 1e-1, 'dims': [20]
    },
    'crime': {
        'batch_size': 128, 'epochs': 60, 'weight_decay': 1e-2, 'dims': [100]
    },
    'health': {
        'batch_size': 256, 'epochs': 80, 'weight_decay': 1e-3, 'dims': [100, 50, 20]
    },
    'lawschool': {
        'batch_size': 128, 'epochs': 100, 'weight_decay': 1e-4, 'dims': [100]
    }
}

if args.with_test:
    p_val = 0.01
    p_test = 0.2
else:
    p_val = 0.2
    p_test = 0.2

for dataset, dataset_params in parameters.items():
    for seed in seeds:
        batch_size = dataset_params['batch_size']
        num_epochs = dataset_params['epochs']
        weight_decay = dataset_params['weight_decay']
        dims = dataset_params['dims']

        cmd = f'python train_clf_baseline.py --dataset {dataset} ' \
              f'--seed {seed} --batch-size {batch_size} ' \
              f'--weight-decay {weight_decay} --num-epochs {num_epochs} ' \
              f'--p_val {p_val} --p_test {p_test} ' \
              f'--classifier-dims {" ".join(map(str, dims))} --load'

        if args.with_test:
            cmd += ' --with-test'

        print(cmd)
        os.system(cmd)
