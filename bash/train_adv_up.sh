python -m train.adversarial_training_clip_up \
--clip_model_name ViT-L-14 \
--pretrained openai \
--dataset imagenet \
--imagenet_root /c/CodesSpring24/Data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
--template std \
--output_normalize False \
--steps 10000 \
--warmup 1400 \
--batch_size 64 \
--loss l2 \
--opt sgd \
--lr 1e-3 \
--wd 1e-5 \
--attack pgd \
--inner_loss l2 \
--norm linf \
--eps 4 \
--iterations_adv 10 \
--stepsize_adv 1 \
--wandb True \
--output_dir /c/CodesSpring24/RobustVLM/cocoadv \
--experiment_name SimCLIP4 \
--log_freq 10 \



# python -m train.adversarial_training_clip_up \
# --clip_model_name ViT-L-14 \
# --pretrained openai \
# --dataset imagenet \
# --imagenet_root /c/CodesSpring24/Data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
# --template std \
# --output_normalize False \
# --steps 10000 \
# --warmup 1400 \
# --batch_size 64 \
# --loss l2 \
# --opt adamw \
# --lr 1e-5 \
# --wd 1e-4 \
# --attack pgd \
# --inner_loss l2 \
# --norm linf \
# --eps 4 \
# --iterations_adv 10 \
# --stepsize_adv 1 \
# --wandb True \
# --output_dir /c/CodesSpring24/RobustVLM/cocoadv \
# --experiment_name SimCLIP4 \
# --log_freq 10 \