#!/usr/bin/env bash
if [ ! -d experiments ]; then
mkdir experiments 
fi

pushd experiments
    #for seed in 5000 10000 15000 20000 25000 30000 35000 40000 45000 
    #for seed in 0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 10000 15000 20000 25000 30000 35000 40000 45000
    for seed in 5500 6000 6500 7000 7500 8000 8500 9000 9500 10500 11000 11500 12000 12500 13000 13500 14000 14500 15500 16000
    do
        for reward_func in mes mean exp_improve
        do
            workdir=nonmyopic_seed${seed}
            if [ ! -d $workdir ]; then mkdir $workdir; fi

            pushd $workdir
            cmd="python ../../nonmyopic_experiments.py ${seed} ${reward_func}"
            echo $cmd
            $cmd
            popd
        done
    done
popd