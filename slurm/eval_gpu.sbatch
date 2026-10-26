#!/bin/bash
#SBATCH -J onepass-eval
#SBATCH -p cluster1-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
cd ~/ALT
. .venv-gpu/bin/activate
export HF_HOME=~/ALT/.cache/hf
python scripts/evaluate.py "$@" --device cuda
