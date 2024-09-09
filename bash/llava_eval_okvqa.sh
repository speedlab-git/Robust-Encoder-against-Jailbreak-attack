#!/bin/bash
# LLaVA evaluation script

python -m vlm_eval.run_evaluation \
--eval_ok_vqa \
--attack none --eps 8 --steps 100 --mask_out none \
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
--ok_vqa_train_image_dir_path /c/CodesSpring24/Data/coco2014/train2014 \
--ok_vqa_train_questions_json_path /c/CodesSpring24/Data/OKVQA/OpenEnded_mscoco_train2014_questions.json \
--ok_vqa_train_annotations_json_path /c/CodesSpring24/Data/OKVQA/mscoco_train2014_annotations.json \
--ok_vqa_test_image_dir_path /c/CodesSpring24/Data/coco2014/val2014 \
--ok_vqa_test_questions_json_path /c/CodesSpring24/Data/OKVQA/OpenEnded_mscoco_val2014_questions.json \
--ok_vqa_test_annotations_json_path /c/CodesSpring24/Data/OKVQA/mscoco_val2014_annotations.json \



python -m vlm_eval.run_evaluation \
--eval_ok_vqa \
--attack none --eps 8 --steps 100 --mask_out none \
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
--ok_vqa_train_image_dir_path /c/CodesSpring24/Data/coco2014/train2014 \
--ok_vqa_train_questions_json_path /c/CodesSpring24/Data/OKVQA/OpenEnded_mscoco_train2014_questions.json \
--ok_vqa_train_annotations_json_path /c/CodesSpring24/Data/OKVQA/mscoco_train2014_annotations.json \
--ok_vqa_test_image_dir_path /c/CodesSpring24/Data/coco2014/val2014 \
--ok_vqa_test_questions_json_path /c/CodesSpring24/Data/OKVQA/OpenEnded_mscoco_val2014_questions.json \
--ok_vqa_test_annotations_json_path /c/CodesSpring24/Data/OKVQA/mscoco_val2014_annotations.json \



python -m vlm_eval.run_evaluation \
--eval_ok_vqa \
--attack none --eps 8 --steps 100 --mask_out none \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/tecoa_eps_4.pt \
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
--ok_vqa_train_image_dir_path /c/CodesSpring24/Data/coco2014/train2014 \
--ok_vqa_train_questions_json_path /c/CodesSpring24/Data/OKVQA/OpenEnded_mscoco_train2014_questions.json \
--ok_vqa_train_annotations_json_path /c/CodesSpring24/Data/OKVQA/mscoco_train2014_annotations.json \
--ok_vqa_test_image_dir_path /c/CodesSpring24/Data/coco2014/val2014 \
--ok_vqa_test_questions_json_path /c/CodesSpring24/Data/OKVQA/OpenEnded_mscoco_val2014_questions.json \
--ok_vqa_test_annotations_json_path /c/CodesSpring24/Data/OKVQA/mscoco_val2014_annotations.json \