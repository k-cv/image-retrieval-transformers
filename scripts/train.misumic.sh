# # 
# python mainc.py \
#   --model deit_small_distilled_patch16_224 \
#   --max-iter 20000 \
#   --dataset misumic \
#   --data-path /home/kfujii/vitruvion/outputs/2024-10-21/14-40-41/features_labels.pth \
#   --rank 1 2 4 8 \
#   --lambda-reg 0.7


# Zero-shot
python new_main.py \
  --model deit_small_distilled_patch16_224 \
  --max-iter 0 \
  --dataset misumic \
  --data-path /home/kfujii/vitruvion/outputs/2024-10-21/14-40-41/features_labels.pth \
  --rank 1 2 4 8 \
  --lambda-reg 0.7 \
  --test-ratio 0.2