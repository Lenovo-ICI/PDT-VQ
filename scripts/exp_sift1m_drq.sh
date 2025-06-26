#!/bin/bash
set -ex

mkdir -p ./log

all_M="4 8 16"

for M in $all_M; do
    python -u train.py --dataset sift1m --data_root data/sift1m --d_hidden 256 --heads 4 --steps 6 --M $M --vq_type drq --trans_type msd --re_rank True --log_path ./log
done
