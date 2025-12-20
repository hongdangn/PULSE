python ./LLaVA/llava/eval/run_llava.py \
    --model-path "/home/dang.nh4/PULSE/output1/fastvlm1.5B/checkpoint-15612" \
    --image-file "/mnt/disk1/backup_user/dang.nh4/ecg_instruct/ecg/code15_v4/2508875-0.png" \
    --query "What are the main features in this ECG image?" \
    --conv-mode "qwen_2" \
    --lora True \
    --device 0
