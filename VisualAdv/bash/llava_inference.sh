#!/bin/bash



# Run the inference script Original CLIP
python -m llava_llama_v2_inference.py \
    --image_file "/c/CodesFall24/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results_llava_llama_v2_constrained_16/bad_prompt_temp_5000.bmp" \
    --output_file llava_inference/results16.jsonl

python get_metric.py --input llava_inference/results16.jsonl --output llava_inference/result_eval16.jsonl

python cal_metrics.py --input llava_inference/result_eval16.jsonl

 # example: git clone git@hf.co:datasets/allenai/c4
