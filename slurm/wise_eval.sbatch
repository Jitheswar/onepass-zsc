#!/bin/bash
#SBATCH -J wise-eval
#SBATCH -p cluster1-cpu
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
# usage: sbatch slurm/wise_eval.sbatch <ckpt> <alpha...>
cd ~/ALT; . .venv/bin/activate; export HF_HOME=~/ALT/.cache/hf
CK=$1; shift
for a in "$@"; do
python scripts/wise.py $CK ${CK}_wise$a $a && python scripts/evaluate.py ${CK}_wise$a --n 100 --threads 32 | grep -E "ckpt=|MEAN|^[a-z0-9_]+ +[0-9]"
done
