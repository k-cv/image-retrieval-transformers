# traiｎ
python new_main.py \
  --model deit_small_distilled_patch16_224 \
  --max-iter 2000 \
  --dataset misumi \
  --rank 1 2 4 8 \
  --lambda-reg 0.7 \
  --test-ratio 0.2 \
  --data-path /home/kfujii/vitruvion/data/images/tmp \
  --geo-features-path /home/kfujii/vitruvion/outputs/2024-12-07/12-39-31/features_paths.pth \
  --seed 0
  # --data-path /home/kfujii/vitruvion/data/images/tmp \
  # --geo-features-path /home/kfujii/vitruvion/outputs/2024-10-21/14-40-41/features_labels.pth \
  # --data-path /home/kfujii/drawing/data \

# Zero-shot
# python new_main.py \
#   --model deit_small_distilled_patch16_224 \
#   --max-iter 0 \
#   --dataset misumi \
#   --data-path /home/kfujii/drawing/data \
#   --geo-features-path /home/kfujii/vitruvion/outputs/2024-10-21/14-40-41/features_labels.pth \
#   --rank 1 2 4 8 \
#   --lambda-reg 0.7 \
#   --test-ratio 0.2