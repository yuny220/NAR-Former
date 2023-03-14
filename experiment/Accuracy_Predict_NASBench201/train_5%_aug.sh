BASE_DIR="/home/disk/NAR-Former"

python $BASE_DIR/main.py \
    --do_train \
    --device 3 \
    --dataset nasbench201 \
    --data_path "$BASE_DIR/data/nasbench201/all_nasbench201.pt" \
    --aug_data_path "$BASE_DIR/data/nasbench201/OurAug.pt" \
    --percent 781 \
    --batch_size 128 \
    --graph_d_model 192 \
    --d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 6 \
    --depths 6 1 1 1 \
    --max_epoch 4000 \
    --model_ema \
    --learning_rate 1e-4 \
    --lambda_diff 0.1 \
    --lambda_consistency 0.5 \
    --save_path "checkpoints_5%/" \
    --embed_type "nerf" \
    --use_extra_token \