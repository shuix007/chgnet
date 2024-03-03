#!/bin/bash
#SBATCH --job-name=pytorch_ddp       # Job name
#SBATCH -p gk
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1          # How many tasks on each node
#SBATCH --gres=gpu:4                 # Number of GPUs per node
#SBATCH --time=01:00:00              # Time limit hrs:min:sec
#SBATCH --output=ddp_%j.log          # Standard output and error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuix007@umn.edu

# Load any modules and activate your Python environment here

# Clear any previously set PyTorch environment variables
unset PYTHONPATH

conda activate Torch2.1
cd /home/karypisg/shuix007/FERMat/chgnet

# Setup environment variables for distributed PyTorch
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=13356

# Run the PyTorch script using torch.distributed.launch or torchrun (for PyTorch >= 1.9)
# srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT your_training_script.py

# echo $MASTER_ADDR:$MASTER_PORT
# echo $SLURM_NTASKS_PER_NODE
# echo $SLURM_NNODES
# echo $WORLD_SIZE
# echo $SLURM_JOB_ID

# If using PyTorch 1.9 or newer, you can also use torchrun (which replaces torch.distributed.launch)
# srun torchrun --nproc_per_node=$SLURM_NTASKS_PER_NODE \
#             --nnodes=$SLURM_NNODES \
#             --max-restarts=1 \
#             --rdzv_id=$SLURM_JOB_ID \
#             --rdzv_backend=c10d \
#             --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#             main.py --submit --distributed --num-nodes 2 --num-gpus 4

srun torchrun --nproc_per_node=4 \
            --nnodes=$SLURM_NNODES \
            --max-restarts=0 \
            --standalone \
            main.py --submit --distributed --num-nodes 1 --num-gpus 4
            # --rdzv_id=$SLURM_JOB_ID \
            # --rdzv_backend=c10d \
            # --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            