EXPID='replace <your checkpoint log id>'

HOST='127.0.0.1'
PORT='3'

NUM_GPU=1

tests=("youtube_Deepfakes" "youtube_FaceSwap" "youtube_Face2Face" "youtube_NeuralTextures")


YOUR_DATA_PATH="<replace your data path>"
YOUR_RESULT_PATH="<replace your result path>"


for test in "${tests[@]}"
do
    echo "Testing on ${test}..."
    CUDA_VISIBLE_DEVICES="0" python test.py \
        --config 'configs/bottleneck_vit_base_patch16_224_spatial.json' \
        --results_path ${YOUR_RESULT_PATH} \
        --test_level 'frame' \
        --data_dir "${YOUR_DATA_PATH}/FaceForensicspp_RECCE" \
        --dataset_name 'FaceForensicspp_RECCE_c23' \
        --dataset_split 'alltype' \
        --test_dataset_name 'FaceForensicspp_RECCE_c23' \
        --test_dataset_split ${test} \
        --launcher pytorch \
        --rank 0 \
        --log_num ${EXPID} \
        --dist-url tcp://${HOST}:2284${PORT} \
        --ffn_adapt \
        --world_size $NUM_GPU
done