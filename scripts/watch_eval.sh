#!/bin/bash
# For each kept checkpoint stepN of a run, evaluate on benchmark (n=300) and dev (n=400) once.
RUN=$1
while true; do
  for d in runs/$RUN/step*; do
    [ -f $d/model.pt ] || continue
    [ -f $d/.queued ] && continue
    sleep 60; touch $d/.queued
    sbatch -J ev-$(basename $d) -p cluster1-cpu -c 32 --mem=48G -t 3:00:00 -o logs/ev-$RUN-$(basename $d)-%j.out --wrap "cd ~/ALT; . .venv/bin/activate; export HF_HOME=~/ALT/.cache/hf; python scripts/evaluate.py $d --n 300 --threads 32 --chunk 16 2>/dev/null | grep -E 'ckpt=|MEAN|^[a-z0-9_]+ +[0-9]'; python scripts/evaluate.py $d --split dev --n 400 --threads 32 --chunk 16 2>/dev/null | grep -E 'ckpt=|MEAN|^[a-z_]+ +[0-9]'"
  done
  grep -q "^done" logs/${2:-onepass-packed2}-*.out 2>/dev/null && break
  sleep 120
done
