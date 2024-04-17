#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shuix007@umn.edu

while getopts m:s: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        s) split=${OPTARG};;
    esac
done

conda activate chgnet-matbench

cd /home/karypisg/shuix007/FERMat/chgnet/

CURRENTEPOCTIME=`date +"%Y-%m-%d-%H-%M-%S"`
python eval_matbench.py --model ${model} --world_size 20 --split ${split} &> ${CURRENTEPOCTIME}.semchgnet.${split}.log.txt