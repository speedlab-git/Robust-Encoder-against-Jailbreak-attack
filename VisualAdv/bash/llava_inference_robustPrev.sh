#!/bin/bash



python -m llava_llama_v2_inference.py \
    --image_file "/c/CodesFall24/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results_llava_llama_v2_constrained_128/bad_prompt.bmp" \
    --output_file llava_inference/results128.jsonl

python get_metric.py --input llava_inference/results128.jsonl --output llava_inference/result_eval128.jsonl

