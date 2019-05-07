#!/usr/bin/env bash
if [ ! -d naive_experiments ]; then
mkdir naive_experiments
fi

pushd naive_experiments
    for seed in {0..10100..100}
    do
        for pathset in dubins
        do
            for goal_only in False 
            do
                for cost in False 
                do
                    for nonmyopic in False True 
                    do
                        for reward_func in naive naive_value #mes mean 
                        do
                          for tree_type in dpw #belief
                          do
                              echo sim_seed${seed}-pathset${pathset}-nonmyopic${nonmyopic}-tree${tree_type}
                              # Dont want to run either cost or goal only
                              if [ ${pathset} = dubins ] && [ ${goal_only} = True ]; then
                                  continue
                              fi
                              if [ ${pathset} = dubins ] && [ ${cost} = True ]; then
                                  continue
                              fi
                              #  Don't want to run myopic MES
                              if [ ${reward_func} = mes ] && [ ${nonmyopic} = False ]; then
                                 continue
                              fi
                              # Myopic; tree_type should be ignored anyways, but should only run once
                              if [ ${nonmyopic} = False ] && [ ${tree_type} = belief ]; then
                                  continue
                              fi

                              if [ ${nonmyopic} = True ] && [ ${reward_func} = mes ] && [ ${tree_type} = belief ]; then
                                 continue
                              fi

                              if [ ${nonmyopic} = True ] && [ ${reward_func} = naive ] && [ ${tree_type} = belief ]; then
                                 continue
                              fi

                              if [ ${nonmyopic} = True ] && [ ${reward_func} = naive_value ] && [ ${tree_type} = belief ]; then
                                 continue
                              fi

                              if [ ${nonmyopic} = True ] && [ ${reward_func} = mean ] && [ ${tree_type} = dpw ]; then
                                 continue
                              fi

                              if [ ${pathset} = fully_reachable_goal ] && [ ${nonmyopic} = True ]; then
                                  continue
                              fi
                              # if [ ${pathset} = fully_reachable_goal ] && [ ${cost} = False ]; then
                              #     continue
                              # fi
                              # if [ ${pathset} = fully_reachable_goal ] && [ ${cost} = False ]; then
                              #     continue
                              # fi

                              if [ ${nonmyopic} = False ]; then
                                workdir=sim_seed${seed}-pathset${pathset}-nonmyopic${nonmyopic}
                              else
                                workdir=sim_seed${seed}-pathset${pathset}-nonmyopic${nonmyopic}-tree${tree_type}
                              fi
                              
                              if [ -d $workdir ] && [ -f ${workdir}/figures/${reward_func}/trajectory-N.SUMMARY.png ] ; then 
                                continue
                              fi

                              if [ -d $workdir ] && [ -f ${workdir}/figures/${reward_func}/trajectory-N.SUMMARY.png ] ; then 
                                continue
                              fi

                              if [ ! -d $workdir ]; then mkdir $workdir; fi

                              pushd $workdir
                              cmd="python ../../nonmyopic_experiments.py ${seed} ${reward_func} ${pathset} ${cost} ${nonmyopic} ${goal_only} ${tree_type}"
                              echo $cmd
                              $cmd
                              popd
                            done
                        done
                    done
                done
            done
        done
    done
popd
