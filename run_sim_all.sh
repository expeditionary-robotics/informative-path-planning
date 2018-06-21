#!/usr/bin/env bash
if [ ! -d experiments ]; then
mkdir experiments
fi

pushd experiments
    #for seed in 5000 10000 15000 20000 25000 30000 35000 40000 45000 
    #for seed in 0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 10000 15000 20000 25000 30000 35000 40000 45000
    #for seed in 5500 6000 6500 7000 7500 8000 8500 9000 9500 10500 11000 11500 12000 12500 13000 13500 14000 14500 15500 16000
    #for seed in 0 100 200 300 400 500 #0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 10000 15000 20000 25000 30000 35000 40000 45000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10500 11000 11500 12000 12500 13000 13500 14000 14500 15500 16000 16500 #17000 17500 18000 18500 19000 19500 20500 21000 21500
    #for seed in 0 100 200 300 400 500 600 700 800 900 1000 1100  #0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 10000 15000 20000 25000 30000 35000 40000 45000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10500 11000 11500 12000 12500 13000 13500 14000 14500 15500 16000 16500 #17000 17500 18000 18500 19000 19500 20500 21000 21500
    #for seed in 300 #0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 10000 15000 20000 25000 30000 35000 40000 45000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10500 11000 11500 12000 12500 13000 13500 14000 14500 15500 16000 16500 #17000 17500 18000 18500 19000 19500 20500 21000 21500
    #for seed in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 #0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 10000 15000 20000 25000 30000 35000 40000 45000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10500 11000 11500 12000 12500 13000 13500 14000 14500 15500 16000 16500 #17000 17500 18000 18500 19000 19500 20500 21000 21500
    for seed in {0..4000..100}
    do
        for pathset in dubins fully_reachable_goal
        do
            for goal_only in False True
            do
                for cost in False  True
                do
                    for nonmyopic in False True 
                    do
                        for reward_func in mes mean 
                        do
                            if [ ${pathset} = dubins ] && [ ${goal_only} = True ]; then
                                continue
                            fi
                            if [ ${pathset} = dubins ] && [ ${cost} = True ]; then
                                continue
                            fi


                            if [ ${pathset} = fully_reachable_goal ] && [ ${nonmyopic} = True ]; then
                                continue
                            fi
                            if [ ${pathset} = fully_reachable_goal ] && [ ${cost} = False ]; then
                                continue
                            fi
                            if [ ${pathset} = fully_reachable_goal ] && [ ${cost} = False ]; then
                                continue
                            fi

                            workdir=sim_seed${seed}-pathset${pathset}-cost${cost}-nonmyopic${nonmyopic}-goal${goal_only}

                            if [ -d $workdir ] && [ -f ${workdir}/figures/${reward_func}/trajectory-N.SUMMARY.png ] ; then 
                              continue
                            fi

                            if [ ! -d $workdir ]; then mkdir $workdir; fi

                            pushd $workdir
                            cmd="python ../../nonmyopic_experiments.py ${seed} ${reward_func} ${pathset} ${cost} ${nonmyopic} ${goal_only}"
                            echo $cmd
                            $cmd
                            popd
                        done
                    done
                done
            done
        done
    done
popd
