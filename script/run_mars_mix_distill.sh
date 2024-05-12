tport=52008
ngpu=1
ROOT=.

CUDA_VISIBLE_DEVICES=4 \
python -m torch.distributed.launch \
    --nproc_per_node=${ngpu} \
    --node_rank=0 \
    --master_port=${tport} \
    $ROOT/train_semi_mars_mix_distill.py \
    --config=/mnt/netdisk/zhangjh/Code/AugSeg-main/exps/pku2mars/config_semi_mix.yaml --seed 2 --port ${tport}

#if test
# python -m torch.distributed.launch --nproc_per_node=1  --node_rank=0  --master_port=52009  train_semi_mars.py --config=/mnt/netdisk/zhangjh/Code/AugSeg-main/exps/pku2mars/config_semi.yaml --seed 2
# --- --- --- 
    # --config=$ROOT/exps/zrun_citys/citys_semi186/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/citys_semi372/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/citys_semi744/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/citys_semi1488/config_semi.yaml --seed 2 --port ${tport}

    # --config=$ROOT/exps/zrun_citys/r50_citys_semi186/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/r50_citys_semi372/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/r50_citys_semi744/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/r50_citys_semi1488/config_semi.yaml --seed 2 --port ${tport}
