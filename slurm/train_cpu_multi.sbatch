#!/bin/bash
#SBATCH -J onepass-cpu
#SBATCH -p cluster1-cpu
#SBATCH --ntasks-per-node=2
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 18:00:00
#SBATCH -o logs/%x-%j.out
# usage: sbatch -N <nodes> slurm/train_cpu_multi.sbatch <run_name> <backbone> [train.py args]
cd ~/ALT
. .venv/bin/activate
export HF_HOME=~/ALT/.cache/hf TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=28
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
RUN=$1; BB=$2; shift 2
echo "nodes: $SLURM_JOB_NODELIST tasks: $SLURM_NTASKS master $MASTER_ADDR"
srun --cpu-bind=sockets python scripts/train.py --out runs/$RUN --backbone $BB --device cpu --threads 28 --workers 3 "$@"
