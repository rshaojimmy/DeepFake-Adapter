EXPID=$(date +"%Y%m%d_%H%M%S")

HOST='127.0.0.1'
PORT='2'

NUM_GPU=2
YOUR_DATA_PATH="<replace your data path>"
YOUR_RESULT_PATH="<replace your result path>"

CUDA_VISIBLE_DEVICES="0,1" python train.py \
    --results_path ${YOUR_RESULT_PATH} \
    --config 'configs/bottleneck_vit_base_patch16_224_spatial.json' \
    --data_dir "${YOUR_DATA_PATH}/FaceForensicspp_RECCE" \
    --dataset_name 'FaceForensicspp_RECCE_c40' \
    --dataset_split "youtube_Deepfakes" \
    --test_dataset_name "youtube_Deepfakes" \
    --launcher pytorch \
    --rank 0 \
    --log_num ${EXPID} \
    --dist-url tcp://${HOST}:2235${PORT} \
    --world_size $NUM_GPU \
    --ffn_adapt \
    --val_epoch 1