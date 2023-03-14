BASE_DIR="/home/disk/NAR-Former"

python $BASE_DIR/main.py \
    --do_train \
    --device 5 \
    --dataset nasbench101 \
    --percent 4236 \
    --data_path "$BASE_DIR/data/nasbench101/all.pt" \
    --batch_size 128 \
    --graph_d_model 192 \
    --d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 6 \
    --depths 6 1 1 1 \
    --max_epoch 6000 \
    --model_ema \
    --learning_rate 1e-4 \
    --lambda_diff 0.1 \
    --save_path "checkpoints_1%_aug/" \
    --embed_type "nerf" \
    --use_extra_token \
    --aug_data_path "$BASE_DIR/data/nasbench101/OurAug.pt" \
    --lambda_consistency 0.5 \