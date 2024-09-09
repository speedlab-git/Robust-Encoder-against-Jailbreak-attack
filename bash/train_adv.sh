python -m train.adversarial_training_clip \
--clip_model_name ViT-L-14 \
--pretrained openai \
--dataset imagenet \
--imagenet_root /c/CodesSpring24/Data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
--template std \
--output_normalize False \
--steps 20000 \
--warmup 1400 \
--batch_size 32 \
--loss l2 \
--opt adamw \
--lr 1e-5 \
--wd 1e-4 \
--attack pgd \
--inner_loss l2 \
--norm linf \
--eps 4 \
--iterations_adv 10 \
--stepsize_adv 1 \
--wandb False \
--output_dir /path/to/out/dir \
--experiment_name FARE4 \
--log_freq 10 \
