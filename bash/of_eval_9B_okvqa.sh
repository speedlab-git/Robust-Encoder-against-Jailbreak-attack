#!/bin/bash

python -m vlm_eval.run_evaluation_of \
--eval_ok_vqa \
--attack apgd --eps 4 --steps 100 --mask_out context \
--vision_encoder_pretrained openai\
--num_samples 200 \
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
--device_n 2 \
--cross_attn_every_n_layers 4 \
--ok_vqa_train_image_dir_path /c/CodesSpring24/Data/coco2014/train2014 \
--ok_vqa_train_questions_json_path /c/CodesSpring24/Data/OKVQA/OpenEnded_mscoco_train2014_questions.json \
--ok_vqa_train_annotations_json_path /c/CodesSpring24/Data/OKVQA/mscoco_train2014_annotations.json \
--ok_vqa_test_image_dir_path /c/CodesSpring24/Data/coco2014/val2014 \
--ok_vqa_test_questions_json_path /c/CodesSpring24/Data/OKVQA/OpenEnded_mscoco_val2014_questions.json \
--ok_vqa_test_annotations_json_path /c/CodesSpring24/Data/OKVQA/mscoco_val2014_annotations.json \


