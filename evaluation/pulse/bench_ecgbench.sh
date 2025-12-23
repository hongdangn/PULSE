SAVE_DIR=./eval_res
CKPT_DIR=/home/dang.nh4/PULSE/ml-fastvlm/output/fastvlm1.5B-orig/checkpoint-1412
# CKPT_DIR="apple/FastVLM-1.5B"
# data_names=(ptb-test code15-test mmmu-ecg cpsc-test g12-test-no-cot csn-test-no-cot ecgqa-test)
# data_names=(code15-test)
data_names=(ptb-test)

#lora
for data_name in "${data_names[@]}"; do
    echo "=========================================================================="
    echo "   [STARTING] Inference for subset: $data_name"
    echo "=========================================================================="

    mkdir -p "${SAVE_DIR}/${data_name}"

    python ./LLaVA/llava/eval/model_ecg_resume.py \
        --model-path $CKPT_DIR \
        --batch_size 36 \
        --model_name "apple/FastVLM-1.5B" \
        --image-folder "./data/ECGBench/images" \
        --question-file "./data/ECGBench/${data_name}.json" \
        --answers-file "${SAVE_DIR}/${data_name}/step-final.jsonl" \
        --conv-mode "qwen_2" \
        --lora True \
        --device 5

done

python evaluation/evaluate_ecgbench.py

#nolora
# for data_name in "${data_names[@]}"; do
#     echo "=========================================================================="
#     echo "   [STARTING] Inference for subset: $data_name"
#     echo "=========================================================================="

#     mkdir -p "${SAVE_DIR}/${data_name}"

#     python ./LLaVA/llava/eval/model_ecg_resume.py \
#         --model-path $CKPT_DIR \
#         --model_name "apple/FastVLM-1.5B" \
#         --batch_size 36 \
#         --image-folder "./data/ECGBench/images" \
#         --question-file "./data/ECGBench/${data_name}.json" \
#         --answers-file "${SAVE_DIR}/${data_name}/step-final.jsonl" \
#         --conv-mode "qwen_2" \
#         --device 5

# done