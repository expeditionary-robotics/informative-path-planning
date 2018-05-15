#!/usr/bin/env bash
if [ ! -d experiments ]; then
mkdir experiments 
fi

pushd experiments
    for seed in 0 500 1000 1500 2000 2500 3000 3500 4000 4500 
    do
        for reward_func in mes mean
        do
            workdir=nonmyopic_seed${seed}
            if [ ! -d $workdir ]; then mkdir $workdir; fi

            pushd $workdir
            cmd="topics.refine.txy --in.words=../../words/words.compressed.texton.csv --in.position=../../words/words.compressed.texton.position.csv --cell.space=32 --retime=1 --alpha=${alpha} --beta=${beta}  -V 500 --gamma=${gamma} --grow.topics.size=1  --threads=8 --online --online.mint=500 "
            cmd="python ../../nonmyopic_experiments.py ${seed} ${reward_func}"
            echo $cmd
            $cmd
            popd
        done
    done
popd
