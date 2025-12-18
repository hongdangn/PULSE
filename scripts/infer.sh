python ./LLaVA/llava/eval/run_llava.py \
    --model-path "PULSE-ECG/PULSE-7B" \
    --image-file "/mnt/disk1/backup_user/dang.nh4/ecg_instruct/ecg/code15_v4/2508875-0.png" \
    --query "What are the main features in this ECG image?" \
    --conv-mode "llava_v1"