#!/bin/bash
# LLaVA evaluation script
python -m vlm_eval.run_evaluation \
--eval_flickr30 \
--attack apgd --eps 8 --steps 100 --mask_out none \
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
--flickr_image_dir_path /c/CodesSpring24/Data/flickr30k_images/flickr30k_images \
--flickr_karpathy_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k.json \
--flickr_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k_coco_style.json \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/ViT-L-14_openai_imagenet_l2_imagenet_SimCLIP4_bVISu/checkpoints/fallback_38600.pt \


python -m vlm_eval.run_evaluation \
--eval_flickr30 \
--attack apgd --eps 8 --steps 100 --mask_out none \
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
--flickr_image_dir_path /c/CodesSpring24/Data/flickr30k_images/flickr30k_images \
--flickr_karpathy_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k.json \
--flickr_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k_coco_style.json \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/fare_eps_4.pt \


python -m vlm_eval.run_evaluation \
--eval_flickr30 \
--attack apgd --eps 8 --steps 100 --mask_out none \
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
--flickr_image_dir_path /c/CodesSpring24/Data/flickr30k_images/flickr30k_images \
--flickr_karpathy_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k.json \
--flickr_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k_coco_style.json \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/tecoa_eps_4.pt \


