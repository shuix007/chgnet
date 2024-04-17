#!/bin/bash -l

# declare -a model_list=("semdedup03_chgnet" "chgnet")
declare -a model_list=("semdedup03_chgnet")

cd /home/karypisg/shuix007/FERMat/chgnet/scripts

for model in ${model_list[@]}; do
    for i in `seq 0 1 1`; do 
        sbatch --gres=gpu:1 --time=24:00:00 -p gk run_matbench.sh -m ${model} -s ${i}
    done
done