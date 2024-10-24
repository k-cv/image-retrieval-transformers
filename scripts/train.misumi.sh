# 通常
# python main.py \
#   --model deit_small_distilled_patch16_224 \
#   --max-iter 2000 \
#   --dataset misumi \
#   --data-path /home/kfujii/vitruvion/data/01_本番 \
#   --rank 1 2 4 8 \
#   --lambda-reg 0.7 \

# Zero-shot
python main.py \
  --model deit_small_distilled_patch16_224 \
  --max-iter 0 \
  --dataset misumi \
  --data-path /home/kfujii/vitruvion/data/01_本番 \
  --rank 1 2 4 8 \
  --lambda-reg 0.7 \
  --test-ratio 0.9