#!/bin/bash

run_name="pretrain_openmath"

python pretrain.py \
    arch=trm \
    data_paths="[data/openmath]" \
    evaluators="[]" \
    epochs=50000 eval_interval=1000 \
    lr=5e-5 puzzle_emb_lr=5e-5 weight_decay=0.1 puzzle_emb_weight_decay=0.1 \
    arch.L_layers=6 \
    arch.H_cycles=4 arch.L_cycles=8 \
    +run_name=${run_name} ema=True