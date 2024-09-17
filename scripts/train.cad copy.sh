# CADデータセット
python mainc.py \
  --model deit_small_distilled_patch16_224 \
  --max-iter 50 \
  --dataset cad \
  --data-path /home/kfujii/vitruvion/outputs/2024-09-05/12-54-06_all_images \
  --rank 1 2 4 8 \
  --lambda-reg 0.7