#!/bin/bash
set -e
# English

models=("aguvis")
model_name_or_path="/cpfs01/data/shared/Group-m6/zeyu.czy/workspace/pythonfile/xlang/junli/workspace/models/1080p/checkpoints/Qwen2-VL-7B-Instruct-sft-stage4.0.11"
for model in "${models[@]}"
do
    python eval_screenspot_pro_parallel.py  \
        --model_type ${model}  \
        --model_name_or_path ${model_name_or_path} \
        --screenspot_imgs "../screenspot-pro-data/images"  \
        --screenspot_test "../screenspot-pro-data/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_dir "./logs" \
        --output_dir "./results/${model}_parallel/" \
        --inst_style "instruction"

done

