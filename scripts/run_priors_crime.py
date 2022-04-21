import os
import numpy as np

batch_size = 300
n_epochs = 300
adv_epochs = 300

for prior in ['autoreg', 'flow', 'gmm']:
    if prior == 'autoreg':
        prior_epochs = 200
    elif prior == 'flow':
        prior_epochs = 300
    else:
        prior_epochs = 0

    for gamma in [0.0001, 0.01, 0.05, 0.1, 0.2]:
        cmd = f'python crime_flow.py --prior {prior} --prior_epochs {prior_epochs} --batch_size {batch_size} --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma {gamma} --with_test --out_file crime/{prior}/gammas/out_{gamma}.txt'
        print(cmd)
        os.system(cmd)


    

