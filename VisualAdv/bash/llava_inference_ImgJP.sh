#!/bin/bash



# Run the inference script Original CLIP
python -m llava_llama_v2_inference_ImgJP \
    --output_file llava_inference/resultsImgjp.jsonl

# python get_metric.py --input llava_inference/results16.jsonl --output llava_inference/result_eval16.jsonl



 # example: git clone git@hf.co:datasets/allenai/c4