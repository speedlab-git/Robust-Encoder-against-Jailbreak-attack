#!/bin/bash

python -m vlm_eval.run_evaluation \
--eval_coco \
--attack apgd --eps 8 --steps 100 --mask_out none \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/ViT-L-14_openai_imagenet_l2_imagenet_SimCLIP4_bVISu/checkpoints/fallback_38600.pt \
--precision float16 \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file llava \
--model llava \
--temperature 0.0 \
--num_beams 1 \
--out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
--model_path liuhaotian/llava-v1.5-7b \
--coco_train_image_dir_path /c/CodesSpring24/Data/coco2014/train2014 \
--coco_val_image_dir_path /c/CodesSpring24/Data/coco2014/val2014 \
--coco_karpathy_json_path eval_benchmark/mscoco_karpathy/karpathy_coco.json \
--coco_annotations_json_path  /c/CodesSpring24/Data/coco2014/annotations/captions_val2014.json



python -m vlm_eval.run_evaluation \
--eval_coco \
--attack apgd --eps 8 --steps 100 --mask_out none \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/fare_eps_4.pt \
--precision float16 \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file llava \
--model llava \
--temperature 0.0 \
--num_beams 1 \
--out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
--model_path liuhaotian/llava-v1.5-7b \
--coco_train_image_dir_path /c/CodesSpring24/Data/coco2014/train2014 \
--coco_val_image_dir_path /c/CodesSpring24/Data/coco2014/val2014 \
--coco_karpathy_json_path eval_benchmark/mscoco_karpathy/karpathy_coco.json \
--coco_annotations_json_path  /c/CodesSpring24/Data/coco2014/annotations/captions_val2014.json


# python -m vlm_eval.run_evaluation \
# --eval_coco \
# --attack apgd --eps 2 --steps 100 --mask_out none \
# --vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/fare_eps_4.pt \
# --precision float16 \
# --num_samples 500 \
# --shots 0 \
# --batch_size 1 \
# --results_file llava \
# --model llava \
# --temperature 0.0 \
# --num_beams 1 \
# --out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
# --model_path liuhaotian/llava-v1.5-7b \
# --coco_train_image_dir_path /c/CodesSpring24/Data/coco2014/train2014 \
# --coco_val_image_dir_path /c/CodesSpring24/Data/coco2014/val2014 \
# --coco_karpathy_json_path eval_benchmark/mscoco_karpathy/karpathy_coco.json \
# --coco_annotations_json_path  /c/CodesSpring24/Data/coco2014/annotations/captions_val2014.json


# python -m vlm_eval.run_evaluation \
# --eval_coco \
# --attack apgd --eps 4 --steps 100 --mask_out none \
# --vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/ViT-L-14_openai_imagenet_l2_imagenet_SimCLIP4_bVISu/checkpoints/fallback_38600.pt \
# --precision float16 \
# --num_samples 500 \
# --shots 0 \
# --batch_size 1 \
# --results_file llava \
# --model llava \
# --temperature 0.0 \
# --num_beams 1 \
# --out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
# --model_path liuhaotian/llava-v1.5-7b \
# --coco_train_image_dir_path /c/CodesSpring24/Data/coco2014/train2014 \
# --coco_val_image_dir_path /c/CodesSpring24/Data/coco2014/val2014 \
# --coco_karpathy_json_path eval_benchmark/mscoco_karpathy/karpathy_coco.json \
# --coco_annotations_json_path  /c/CodesSpring24/Data/coco2014/annotations/captions_val2014.json


# python -m vlm_eval.run_evaluation \
# --eval_coco \
# --attack apgd --eps 8 --steps 100 --mask_out none \
# --vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/fare_eps_4.pt \
# --precision float16 \
# --num_samples 500 \
# --shots 0 \
# --batch_size 1 \
# --results_file llava \
# --model llava \
# --temperature 0.0 \
# --num_beams 1 \
# --out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
# --model_path liuhaotian/llava-v1.5-7b \
# --coco_train_image_dir_path /c/CodesSpring24/Data/coco2014/train2014 \
# --coco_val_image_dir_path /c/CodesSpring24/Data/coco2014/val2014 \
# --coco_karpathy_json_path eval_benchmark/mscoco_karpathy/karpathy_coco.json \
# --coco_annotations_json_path  /c/CodesSpring24/Data/coco2014/annotations/captions_val2014.json

