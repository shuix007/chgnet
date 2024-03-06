#!/bin/bash
#SBATCH --job-name=pytorch_ddp       # Job name
#SBATCH -p gk
#SBATCH --nodes=6                    # Number of nodes
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1          # How many tasks on each node
#SBATCH --gres=gpu:4                 # Number of GPUs per node
#SBATCH --time=8:00:00              # Time limit hrs:min:sec
#SBATCH --output=ddp_%j.log          # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuix007@umn.edu

unset PYTHONPATH

conda activate Torch2.1
cd /home/karypisg/shuix007/FERMat/chgnet

# Setup environment variables for distributed PyTorch
export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=13356

srun torchrun --nproc_per_node=4 \
            --nnodes=$SLURM_NNODES \
            --max-restarts=0 \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            main.py \
            --submit \
            --distributed \
            --num-nodes $SLURM_NNODES \
            --num-gpus 4 \
            --json_filename data/MPtrj_2022.9_full.json \
            --identifier MPtraj_full_CL 
            