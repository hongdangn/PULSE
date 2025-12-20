SAVE_DIR=./eval_res
CKPT_DIR=/home/dang.nh4/PULSE/output/fastvlm1.5B/checkpoint-98876
data_names=(ptb-test code15-test mmmu-ecg cpsc-test g12-test-no-cot csn-test-no-cot ecgqa-test)

for data_name in "${data_names[@]}"; do
    echo "=========================================================================="
    echo "   [STARTING] Inference for subset: $data_name"
    echo "=========================================================================="

    mkdir -p "${SAVE_DIR}/${data_name}"

    python ./LLaVA/llava/eval/model_ecg_resume.py \
        --model-path $CKPT_DIR \
        --model_name "apple/FastVLM-1.5B" \
        --image-folder "./data/ECGBench/images" \
        --question-file "./data/ECGBench/${data_name}.json" \
        --answers-file "${SAVE_DIR}/${data_name}/step-final.jsonl" \
        --conv-mode "qwen_2" \
        --lora True \
        --device 3

done