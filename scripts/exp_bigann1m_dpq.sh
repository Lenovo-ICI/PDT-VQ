#!/bin/bash
set -ex

mkdir -p ./log

all_M="4 8 16"

for M in $all_M; do
    python -u train.py --dataset bigann1m --data_root data/bigann --d 128 --d_hidden 256 --heads 4 --steps 12 --M $M --vq_type dpq --trans_type msd --re_rank True --log_path ./log
done


