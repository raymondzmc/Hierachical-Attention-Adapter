#!/bin/bash
#SBATCH --gres=gpu:a100l:8       # Request GPU "generic resources"
#SBATCH --cpus-per-task=32  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=192000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=2-0:00
#SBATCH --output=outfile.out
module load StdEnv/2023 arrow/15 python/3.10
source ~/projects/def-carenini/liraymo6/envs/project/bin/activate

