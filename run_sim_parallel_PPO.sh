#!/bin/bash

n_steps_s="25 50 100 250 500"

for n_steps in $n_steps_s; do
    python Run_LHZ_QAOA_PPO.py --n_steps "$n_steps" &> data/progress/n_steps_"${n_steps}".txt &
done;
