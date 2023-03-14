BASE_DIR="/home/disk/NAR-Former"

for PRETRAINED in "nasbench101_latest" "nasbench101_model_best" "nasbench101_model_best_ema"; do

python $BASE_DIR/main.py \
    --dataset nasbench101 \
    --percent 21180 \
    --data_path "$BASE_DIR/data/nasbench101/all.pt" \
    --batch_size 2048 \
    --graph_d_model 192 \
    --d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 6 \
    --depths 6 1 1 1 \
    --lambda_diff 0.1 \
    --save_path "checkpoints_5%_aug/${PRETRAINED}_test_all" \
    --pretrained_path "checkpoints_5%_aug/${PRETRAINED}.pth.tar" \
    --embed_type "nerf" \
    --use_extra_token \
    --aug_data_path "$BASE_DIR/data/nasbench101/OurAug.pt" \
    --lambda_consistency 0.5 \

done