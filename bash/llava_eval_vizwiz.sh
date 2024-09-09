#!/bin/bash
# LLaVA evaluation script
python -m vlm_eval.run_evaluation \
--eval_vizwiz \
--attack none --eps 2 --steps 100 --mask_out none \
--vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/fare_eps_4.pt \
--precision float16 \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file llava \
--model llava \
--temperature 0.0 \
--num_beams 1 \
--device_n 2 \
--out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
--model_path liuhaotian/llava-v1.5-7b \
--vizwiz_train_image_dir_path /c/CodesSpring24/Data/archive/train/train \
--vizwiz_test_image_dir_path /c/CodesSpring24/Data/archive/val/val \
--vizwiz_train_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_annotations_vqa_format.json


# python -m vlm_eval.run_evaluation \
# --eval_vizwiz \
# --attack none --eps 2 --steps 100 --mask_out none \
# --vision_encoder_pretrained openai \
# --precision float16 \
# --num_samples 500 \
# --shots 0 \
# --batch_size 1 \
# --results_file llava \
# --model llava \
# --temperature 0.0 \
# --num_beams 1 \
# --device_n 2 \
# --out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
# --model_path liuhaotian/llava-v1.5-7b \
# --vizwiz_train_image_dir_path /c/CodesSpring24/Data/archive/train/train \
# --vizwiz_test_image_dir_path /c/CodesSpring24/Data/archive/val/val \
# --vizwiz_train_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_questions_vqa_format.json \
# --vizwiz_train_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
# --vizwiz_test_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_questions_vqa_format.json \
# --vizwiz_test_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_annotations_vqa_format.json




# python -m vlm_eval.run_evaluation \
# --eval_vizwiz \
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
# --device_n 2 \
# --out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
# --model_path liuhaotian/llava-v1.5-7b \
# --vizwiz_train_image_dir_path /c/CodesSpring24/Data/archive/train/train \
# --vizwiz_test_image_dir_path /c/CodesSpring24/Data/archive/val/val \
# --vizwiz_train_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_questions_vqa_format.json \
# --vizwiz_train_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
# --vizwiz_test_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_questions_vqa_format.json \
# --vizwiz_test_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_annotations_vqa_format.json


# python -m vlm_eval.run_evaluation \
# --eval_vizwiz \
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
# --device_n 2 \
# --out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
# --model_path liuhaotian/llava-v1.5-7b \
# --vizwiz_train_image_dir_path /c/CodesSpring24/Data/archive/train/train \
# --vizwiz_test_image_dir_path /c/CodesSpring24/Data/archive/val/val \
# --vizwiz_train_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_questions_vqa_format.json \
# --vizwiz_train_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
# --vizwiz_test_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_questions_vqa_format.json \
# --vizwiz_test_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_annotations_vqa_format.json



# python -m vlm_eval.run_evaluation \
# --eval_vizwiz \
# --attack apgd --eps 2 --steps 100 --mask_out none \
# --vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/ViT-L-14_openai_imagenet_l2_imagenet_SimRClip4/checkpoints/step_20000.pt \
# --precision float16 \
# --num_samples 500 \
# --shots 0 \
# --batch_size 1 \
# --results_file llava \
# --model llava \
# --temperature 0.0 \
# --num_beams 1 \
# --device_n 2 \
# --out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
# --model_path liuhaotian/llava-v1.5-7b \
# --vizwiz_train_image_dir_path /c/CodesSpring24/Data/archive/train/train \
# --vizwiz_test_image_dir_path /c/CodesSpring24/Data/archive/val/val \
# --vizwiz_train_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_questions_vqa_format.json \
# --vizwiz_train_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
# --vizwiz_test_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_questions_vqa_format.json \
# --vizwiz_test_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_annotations_vqa_format.json


# python -m vlm_eval.run_evaluation \
# --eval_vizwiz \
# --attack none --eps 2 --steps 100 --mask_out none \
# --vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/ViT-L-14_openai_imagenet_l2_imagenet_SimRClip4/checkpoints/step_20000.pt \
# --precision float16 \
# --num_samples 500 \
# --shots 0 \
# --batch_size 1 \
# --results_file llava \
# --model llava \
# --temperature 0.0 \
# --num_beams 1 \
# --device_n 2 \
# --out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
# --model_path liuhaotian/llava-v1.5-7b \
# --vizwiz_train_image_dir_path /c/CodesSpring24/Data/archive/train/train \
# --vizwiz_test_image_dir_path /c/CodesSpring24/Data/archive/val/val \
# --vizwiz_train_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_questions_vqa_format.json \
# --vizwiz_train_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
# --vizwiz_test_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_questions_vqa_format.json \
# --vizwiz_test_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_annotations_vqa_format.json


# python -m vlm_eval.run_evaluation \
# --eval_vizwiz \
# --attack apgd --eps 4 --steps 100 --mask_out none \
# --vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/ViT-L-14_openai_imagenet_l2_imagenet_SimRClip4/checkpoints/step_20000.pt \
# --precision float16 \
# --num_samples 500 \
# --shots 0 \
# --batch_size 1 \
# --results_file llava \
# --model llava \
# --temperature 0.0 \
# --num_beams 1 \
# --device_n 2 \
# --out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
# --model_path liuhaotian/llava-v1.5-7b \
# --vizwiz_train_image_dir_path /c/CodesSpring24/Data/archive/train/train \
# --vizwiz_test_image_dir_path /c/CodesSpring24/Data/archive/val/val \
# --vizwiz_train_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_questions_vqa_format.json \
# --vizwiz_train_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
# --vizwiz_test_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_questions_vqa_format.json \
# --vizwiz_test_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_annotations_vqa_format.json


# python -m vlm_eval.run_evaluation \
# --eval_vizwiz \
# --attack apgd --eps 4 --steps 100 --mask_out none \
# --vision_encoder_pretrained /c/CodesSpring24/RobustVLM/cocoadv/fare_eps_4.pt \
# --precision float16 \
# --num_samples 500 \
# --shots 0 \
# --batch_size 1 \
# --results_file llava \
# --model llava \
# --temperature 0.0 \
# --num_beams 1 \
# --device_n 2 \
# --out_base_path /c/CodesSpring24/RobustVLM/cocoadv \
# --model_path liuhaotian/llava-v1.5-7b \
# --vizwiz_train_image_dir_path /c/CodesSpring24/Data/archive/train/train \
# --vizwiz_test_image_dir_path /c/CodesSpring24/Data/archive/val/val \
# --vizwiz_train_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_questions_vqa_format.json \
# --vizwiz_train_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/train_annotations_vqa_format.json \
# --vizwiz_test_questions_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_questions_vqa_format.json \
# --vizwiz_test_annotations_json_path /c/CodesSpring24/RobustVLM/eval_benchmark/vizwiz/val_annotations_vqa_format.json
