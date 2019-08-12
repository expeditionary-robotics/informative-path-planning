#!/usr/bin/env bash
if [ ! -d thesis_swell3d_experiments ]; then
mkdir thesis_swell3d_experiments
fi

pushd thesis_swell3d_experiments
    for seed in {100..1000..100}
    do
        for nonmyopic in False True
        do
            for reward_func in gumbel mean #exp_improve
            do
              for tree_type in dpw belief
              do
                  echo sim_seed${seed}-nonmyopic${nonmyopic}-tree${tree_type}
                  if [ ${nonmyopic} = True ] && [ ${reward_func} = gumbel ] && [ ${tree_type} = belief ]; then
                    continue
                  fi
                  if [ ${nonmyopic} = True ] && [ ${reward_func} = mean ] && [ ${tree_type} = dpw ]; then
                    continue
                  fi

                  if [ ${nonmyopic} = False ]; then
                    workdir=sim_seed${seed}-nonmyopic${nonmyopic}
                  else
                    workdir=sim_seed${seed}-nonmyopic${nonmyopic}-tree${tree_type}
                  fi
                  
                  if [ -d $workdir ] && [ -f ${workdir}/figures/${reward_func}/trajectory-N.SUMMARY.png ] ; then 
                    continue
                  fi

                  if [ -d $workdir ] && [ -f ${workdir}/figures/${reward_func}/trajectory-N.SUMMARY.png ] ; then 
                    continue
                  fi

                  if [ ! -d $workdir ]; then mkdir $workdir; fi

                  pushd $workdir
                  if [ ${nonmyopic} = False ]; then
                    cmd="python ../../../src/myopic_simulations.py -s ${seed} -r ${reward_func} -t ${tree_type}"
                  else
                    cmd="python ../../../src/myopic_simulations.py -s ${seed} -r ${reward_func} -n -t ${tree_type}"
                  fi
                  echo $cmd
                  $cmd
                  popd
                done
            done
        done
    done
popd
