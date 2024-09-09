#!/bin/bash
python -m vlm_eval.run_evaluation_of \
--eval_flickr30 \
--attack apgd --eps 4 --steps 100 --mask_out context \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file res9B \
--model open_flamingo \
--out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
--vision_encoder_path ViT-L-14 \
--checkpoint_path /c/CodesSpring24/RobustVLM/flamingocheckpoint.pt \
--lm_path anas-awadalla/mpt-7b \
--lm_tokenizer_path anas-awadalla/mpt-7b \
--precision float16 \
--cross_attn_every_n_layers 4 \
--flickr_image_dir_path /c/CodesSpring24/Data/flickr30k_images/flickr30k_images \
--flickr_karpathy_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k.json \
--flickr_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k_coco_style.json \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/ViT-L-14_openai_imagenet_l2_imagenet_SimCLIP4_bVISu/checkpoints/fallback_38600.pt \

python -m vlm_eval.run_evaluation_of \
--eval_flickr30 \
--attack apgd --eps 4 --steps 100 --mask_out context \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file res9B \
--model open_flamingo \
--out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
--vision_encoder_path ViT-L-14 \
--checkpoint_path /c/CodesSpring24/RobustVLM/flamingocheckpoint.pt \
--lm_path anas-awadalla/mpt-7b \
--lm_tokenizer_path anas-awadalla/mpt-7b \
--precision float16 \
--cross_attn_every_n_layers 4 \
--flickr_image_dir_path /c/CodesSpring24/Data/flickr30k_images/flickr30k_images \
--flickr_karpathy_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k.json \
--flickr_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k_coco_style.json \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/fare_eps_4.pt \



python -m vlm_eval.run_evaluation_of \
--eval_flickr30 \
--attack apgd --eps 4 --steps 100 --mask_out context \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file res9B \
--model open_flamingo \
--out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
--vision_encoder_path ViT-L-14 \
--checkpoint_path /c/CodesSpring24/RobustVLM/flamingocheckpoint.pt \
--lm_path anas-awadalla/mpt-7b \
--lm_tokenizer_path anas-awadalla/mpt-7b \
--precision float16 \
--cross_attn_every_n_layers 4 \
--flickr_image_dir_path /c/CodesSpring24/Data/flickr30k_images/flickr30k_images \
--flickr_karpathy_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k.json \
--flickr_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/flickr30k/dataset_flickr30k_coco_style.json \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/ViT-L-14_openai_imagenet_l2_imagenet_SimRClip4/checkpoints/fallback_20000.pt \
