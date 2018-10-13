#!/usr/bin/env bash
if [ ! -d experiments ]; then
mkdir experiments 
fi

pushd experiments
    for seed in 300 350 400 450
    do
        for reward_func in exp_improve mean mes
        do
            workdir=nonmyopic_seed_fs${seed}
            if [ ! -d $workdir ]; then mkdir $workdir; fi

            pushd $workdir
            cmd="python ../../nonmyopic_experiments.py ${seed} ${reward_func}"
            echo $cmd
            $cmd
            popd
        done
    done
popd
