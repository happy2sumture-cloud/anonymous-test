CUDA_VISIBLE_DEVICES=0,1 python imu_runner.py train \
  --arch transformer \
  --train_csv /home/sduu1/userspace-20T-2/xiaoxiaofang/code/imu/dataset_v1/splits_user/train.csv \
  --val_csv   /home/sduu1/userspace-20T-2/xiaoxiaofang/code/imu/dataset_v1/splits_user/val.csv \
  --label_map /home/sduu1/userspace-20T-2/xiaoxiaofang/code/imu/dataset_v1/label_map.json \
  --batch_size 32 --epochs 250 --augment \
  --ckpt_out /home/sduu1/userspace-20T-2/xiaoxiaofang/code/imu/net/transformer_output/transformer_best_ck.pt

# screen -dmS imu_train_bilstm_attn \
# bash -lc 'CUDA_VISIBLE_DEVICES=1 python imu_runner.py train \
#   --arch bilstm_attn \
#   --train_csv /home/sduu2/userspace-18T-1/xxf/slr/imu/dataset_v1/splits_user/train.csv \
#   --val_csv   /home/sduu2/userspace-18T-1/xxf/slr/imu/dataset_v1/splits_user/val.csv \
#   --label_map /home/sduu2/userspace-18T-1/xxf/slr/imu/dataset_v1/label_map.json \
#   --batch_size 32 --epochs 250 --augment \
#   --ckpt_out /home/sduu2/userspace-18T-1/xxf/slr/imu/net/bilstm_attn_output/bilstm_attn_best_ck.pt'


# screen -dmS imu_train_bilstm_attn \
# bash -lc 'set -e
# OUTDIR=/home/sduu2/userspace-18T-1/xxf/slr/imu/net/bilstm_attn_output
# mkdir -p "$OUTDIR"
# cd /home/sduu2/userspace-18T-1/xxf/slr/imu/net && \
# CUDA_VISIBLE_DEVICES=1 python imu_runner.py train \
#   --arch bilstm_attn \
#   --train_csv /home/sduu2/userspace-18T-1/xxf/slr/imu/dataset_v1/splits_user/train.csv \
#   --val_csv   /home/sduu2/userspace-18T-1/xxf/slr/imu/dataset_v1/splits_user/val.csv \
#   --label_map /home/sduu2/userspace-18T-1/xxf/slr/imu/dataset_v1/label_map.json \
#   --batch_size 32 --epochs 250 --augment \
#   --ckpt_out "$OUTDIR/bilstm_attn_best_ck.pt" \
#   2>&1 | tee -a "$OUTDIR/train_$(date +%F_%H-%M-%S).log"'

