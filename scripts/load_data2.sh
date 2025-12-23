BASE_URL="https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/resolve/main"
data_names=(code15_v4 ptb-xl)
output_path="/mnt/disk1/backup_user/dang.nh4/ecg_instruct/ecg"

for SUBSET_NAME in "${data_names[@]}"; do
    mkdir -p "${output_path}/${SUBSET_NAME}"
done

for i in $(seq 16 30); do
    TAR_FILE="shard_${i}.tar.gz"

    wget "${BASE_URL}/${TAR_FILE}" -O "${output_path}/$TAR_FILE"

    mkdir -p "${output_path}/shard_${i}"
    tar -xzf "${output_path}/$TAR_FILE" -C "${output_path}/shard_${i}"
    rm -f "${output_path}/$TAR_FILE"

    for SUBSET_NAME in "${data_names[@]}"; do
        DEST_DIR="${output_path}/${SUBSET_NAME}"
        if [ -d "${output_path}/shard_${i}/${SUBSET_NAME}" ]; then
            mv "${output_path}/shard_${i}/${SUBSET_NAME}/"* "${DEST_DIR}"
        fi
    done

    rm -rf "${output_path}/shard_${i}"
done
