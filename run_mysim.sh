#!/usr/bin/env bash
if [ ! -d experiments ]; then
mkdir experiments 
fi

pushd experiments
    for seed in 0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 #10000 15000 20000 25000 30000 35000 40000 45000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10500 11000 11500 12000 12500 13000 13500 14000 14500 15500 16000 16500 #17000 17500 18000 18500 19000 19500 20500 21000 21500
    do
        for reward_func in exp_improve mean mes
        do
            workdir=myopic_seed${seed}
            if [ ! -d $workdir ]; then mkdir $workdir; fi

            pushd $workdir
            cmd="python ../../myopic_experiments.py ${seed} ${reward_func}"
            echo $cmd
            $cmd
            popd
        done
    done
popd
