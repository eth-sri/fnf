#!/bin/bash

rm res_fair_flow.txt;
touch res_fair_flow.txt
for seed in {0..99};
do
	echo ${seed}
	python fair_flow_multi.py --n_epochs 300 --adv_epochs 100 --n_flows 1 --log_epochs 1000 --gamma 0.5 --seed ${seed} --batch_size 256 --kl_start 0 --kl_end 150 >> res_fair_flow.txt
done
