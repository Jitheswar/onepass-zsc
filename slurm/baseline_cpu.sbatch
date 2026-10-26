#!/bin/bash
#SBATCH -J zs-baseline
#SBATCH -p cluster1-cpu
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
# usage: sbatch slurm/baseline_cpu.sbatch <script.py> <model> [args]
cd ~/ALT
. .venv/bin/activate
export HF_HOME=~/ALT/.cache/hf
S=$1; shift
python scripts/$S "$@" --threads 32
