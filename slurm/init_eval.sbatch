#!/bin/bash
#SBATCH -J init-eval
#SBATCH -p cluster1-cpu
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
cd ~/ALT; . .venv/bin/activate; export HF_HOME=~/ALT/.cache/hf
python scripts/save_init.py MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c runs/init_nli --nli_head && \
python scripts/evaluate.py runs/init_nli --n 100 --threads 32
