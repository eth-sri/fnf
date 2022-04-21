import os
import numpy as np

prior_epochs = 200
n_epochs = 150
adv_epochs = 300
batch_size = 300

for gamma in [0, 0.99]:
    # Default label command 
    default_cmd = f'python health_flow.py --load --batch_size {batch_size} --prior_epochs {prior_epochs} --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma {gamma} --out_file health/opp_None_transfer_True/out_default_{gamma}.txt --transfer --with_test'
    os.system(default_cmd)

    labels = ['MSC2a3', 'METAB3']
    for label in labels:
        cmd = f'python health_flow.py --load --batch_size {batch_size} --prior_epochs 0 --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma 0 --out_file health/opp_None_transfer_True/out_{label}_{gamma}.txt --transfer --load_prior --load_enc --label PrimaryConditionGroup={label} --with_test'
        print(cmd)
        os.system(cmd)
