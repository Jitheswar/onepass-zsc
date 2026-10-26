#!/bin/bash
#SBATCH -J onepass-eval
#SBATCH -p cluster1-cpu
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o logs/%x-%j.out
# usage: sbatch slurm/eval_cpu.sbatch <ckpt_dir> [evaluate.py args]
cd ~/ALT
. .venv/bin/activate
export HF_HOME=~/ALT/.cache/hf
python scripts/evaluate.py "$@" --threads 32 --device cpu
