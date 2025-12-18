wget https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/resolve/main/ECGInstruct.json
mv ECGInstruct.json /mnt/disk1/backup_user/dang.nh4/ecg_instruct/ECGInstruct.json

wget https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/resolve/main/shard_1.tar.gz
mv shard_1.tar.gz /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_1.tar.gz
tar -xzf /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_1.tar.gz -C /mnt/disk1/backup_user/dang.nh4/ecg_instruct/ecg

wget https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/resolve/main/shard_2.tar.gz
mv shard_2.tar.gz /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_2.tar.gz
tar -xzf /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_2.tar.gz -C /mnt/disk1/backup_user/dang.nh4/ecg_instruct/ecg

wget https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/resolve/main/shard_10.tar.gz
mv shard_10.tar.gz /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_10.tar.gz
tar -xzf /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_10.tar.gz -C /mnt/disk1/backup_user/dang.nh4/ecg_instruct/ecg

wget https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/resolve/main/shard_11.tar.gz
mv shard_11.tar.gz /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_11.tar.gz
tar -xzf /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_11.tar.gz -C /mnt/disk1/backup_user/dang.nh4/ecg_instruct/ecg

wget https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/resolve/main/shard_20.tar.gz 
mv shard_20.tar.gz /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_20.tar.gz
tar -xzf /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_20.tar.gz -C /mnt/disk1/backup_user/dang.nh4/ecg_instruct/ecg

wget https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/resolve/main/shard_21.tar.gz
mv shard_21.tar.gz /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_21.tar.gz
tar -xzf /mnt/disk1/backup_user/dang.nh4/ecg_instruct/shard_21.tar.gz -C /mnt/disk1/backup_user/dang.nh4/ecg_instruct/ecg

# wget -c -P /mnt/disk1/backup_user/dang.nh4/ecg_instruct https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/resolve/main/shard_21.tar.gz