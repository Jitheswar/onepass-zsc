#!/bin/bash
#SBATCH -J init-eval2
#SBATCH -p cluster1-cpu
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
cd ~/ALT; . .venv/bin/activate; export HF_HOME=~/ALT/.cache/hf
BB=MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c
for v in "tf --text_first" "tft --text_first --template"; do set -- $v; n=$1; shift
python scripts/save_init.py $BB runs/init_$n --nli_head "$@" && python scripts/evaluate.py runs/init_$n --n 100 --threads 32 --max_len 512 --chunk 16 | grep -E "ckpt=|MEAN|^[a-z0-9_]+ +[0-9]"
done
