#!/usr/bin/env bash
if [ ! -d spacetime_experiments ]; then
mkdir spacetime_experiments
fi

pushd spacetime_experiments
    for seed in {250..10000..100}
    do
        for pathset in dubins
        do
            for goal_only in False 
            do
                for cost in False 
                do
                    for nonmyopic in True False 
                    do
                        for reward_func in mes mean naive naive_value #mes #mean 
                        do
                          for tree_type in dpw belief
                          do
                              echo sim_seed${seed}-pathset${pathset}-nonmyopic${nonmyopic}-tree${tree_type}
                              # Dont want to run either cost or goal only
                              #  Don't want to run myopic UCB
                              #if [ ${reward_func} = mean ] && [ ${nonmyopic} = False ]; then
                              #    continue
                              #fi
                              # Myopic; tree_type should be ignored anyways, but should only run once
                              if [ ${nonmyopic} = False ] && [ ${tree_type} = belief ]; then
                                  continue
                              fi

                              if [ ${nonmyopic} = True ] && [ ${reward_func} = naive ] && [ ${tree_type} = belief ]; then
                                 continue
                              fi

                              if [ ${nonmyopic} = True ] && [ ${reward_func} = naive_value ] && [ ${tree_type} = belief ]; then
                                 continue
                              fi
                              if [ ${nonmyopic} = True ] && [ ${reward_func} = mes ] && [ ${tree_type} = belief ]; then
                                 continue
                              fi
                              if [ ${nonmyopic} = True ] && [ ${reward_func} = mean ] && [ ${tree_type} = dpw ]; then
                                 continue
                              fi

                              if [ ${nonmyopic} = False ]; then
                                workdir=sim_seed${seed}-pathset${pathset}-nonmyopic${nonmyopic}
                              else
                                workdir=sim_seed${seed}-pathset${pathset}-nonmyopic${nonmyopic}-tree${tree_type}
                              fi
                              
                              if [ -d $workdir ] && [ -f ${workdir}/figures/${reward_func}/trajectory-N.SUMMARY.png ] ; then 
                                continue
                              fi

                              if [ ! -d $workdir ]; then mkdir $workdir; fi

                              pushd $workdir
                              if [ ${nonmyopic} = False ]; then
                                cmd="python ../../nonmyopic_experiments.py -s ${seed} -r ${reward_func} -p ${pathset}"
                              fi

                              if [ ${nonmyopic} = True ]; then 
                                cmd="python ../../nonmyopic_experiments.py -s ${seed} -r ${reward_func} -p ${pathset} -n -t ${tree_type}"
                              fi

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
