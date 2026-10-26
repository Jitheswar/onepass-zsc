#!/bin/bash
#SBATCH -J onepass-train
#SBATCH -p cluster1-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x-%j.out
# usage: sbatch slurm/train_gpu.sbatch <run_name> <backbone> [extra train.py args...]
cd ~/ALT
. .venv-gpu/bin/activate
export HF_HOME=~/ALT/.cache/hf TOKENIZERS_PARALLELISM=false
RUN=$1; BB=$2; shift 2
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/train.py --out runs/$RUN --backbone $BB "$@"
