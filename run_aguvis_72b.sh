#!/bin/bash
set -e
# English

models=("aguvis_72b")
for model in "${models[@]}"
do
    python eval_screenspot_pro_concurrent.py  \
        --model_type ${model}  \
        --screenspot_imgs "../screenspot-pro-data/images"  \
        --screenspot_test "../screenspot-pro-data/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results/${model}.json" \
        --inst_style "instruction"

done

