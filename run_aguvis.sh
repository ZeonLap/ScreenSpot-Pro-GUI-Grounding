#!/bin/bash
set -e
# English

models=("aguvis")
model_name_or_path="/cpfs01/data/shared/Group-m6/zeyu.czy/workspace/pythonfile/xlang/junli/workspace/models/Qwen2-VL-7B-Instruct-sft-stage5.0-5.0.0_lr1e_5-bsz128-from4.0.11/"
for model in "${models[@]}"
do
    python eval_screenspot_pro.py  \
        --model_type ${model}  \
        --model_name_or_path ${model_name_or_path} \
        --screenspot_imgs "../data/ScreenSpot-Pro/images"  \
        --screenspot_test "../data/ScreenSpot-Pro/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results/${model}.json" \
        --inst_style "instruction"

done

