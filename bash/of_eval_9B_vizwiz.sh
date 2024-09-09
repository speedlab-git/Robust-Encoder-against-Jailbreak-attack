#!/bin/bash

python -m vlm_eval.run_evaluation_of \
--eval_vizwiz \
--attack apgd --eps 2 --steps 100 --mask_out context \
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
--vizwiz_train_image_dir_path /c/CodesSpring24/Data/archive/train/train \
--vizwiz_test_image_dir_path /c/CodesSpring24/Data/archive/val/val \
--vizwiz_train_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_annotations_vqa_format.json \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/ViT-L-14_openai_imagenet_l2_imagenet_SimCLIP4_bVISu/checkpoints/fallback_38600.pt \

python -m vlm_eval.run_evaluation_of \
--eval_vizwiz \
--attack apgd --eps 2 --steps 100 --mask_out context \
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
--vizwiz_train_image_dir_path /c/CodesSpring24/Data/archive/train/train \
--vizwiz_test_image_dir_path /c/CodesSpring24/Data/archive/val/val \
--vizwiz_train_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_annotations_vqa_format.json \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/fare_eps_4.pt \



# --vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/tecoa_eps_4.pt \
