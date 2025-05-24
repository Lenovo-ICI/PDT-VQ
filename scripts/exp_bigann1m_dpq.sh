set -ex

mkdir -p ./experiments/exp_bigann1m_dpq/

# nohup sh scripts/exp_bigann1m_dpq.sh >./experiments/bigann1m_dpq.out &
all_M="16"
all_heads="1 2 3 4"
all_steps="2 4 6 8 10 12"

dataset=bigann1m
data_root=/run/wwk/datasets
d=128
d_hidden=256
re_rank=True
log_path=experiments/exp_bigann1m_dpq
vq_type=dpq
trans_type=msd

for M in $all_M; do
    for heads in $all_heads; do
        for steps in $all_steps; do
            python -u train.py --dataset $dataset --data_root $data_root --d $d --d_hidden $d_hidden --heads $heads --steps $steps --M $M --vq_type $vq_type --trans_type $trans_type --re_rank $re_rank --log_path $log_path
        done
    done
done


