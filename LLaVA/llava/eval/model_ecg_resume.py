import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from peft import LoraConfig, PeftModel
import time
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_module_state(model, module_path):
    model_state = model.state_dict()
    module_state = torch.load(module_path)
    update_state = {k: v for k, v in module_state.items() if k in model_state}

    num_loaded_params = sum(p.numel() for n, p in update_state.items())
    print(f"The number of loaded parameters from {module_path}: {num_loaded_params}")

    model_state.update(update_state)
    model.load_state_dict(model_state)


def eval_model(args):
    # Model
    disable_torch_init()
    # model_path = args.model_path
    model_name = args.model_name
    # model_name = get_model_name_from_path(model_path)
    device = torch.device("cuda:" + str(args.device))
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, 
                                                                           args.model_base, 
                                                                           model_name,
                                                                           lora=args.lora, 
                                                                           device_map=device)

    if args.lora:
        print(f'Loading Pre-trained LoRA model from {args.model_path}')
        lora_config = LoraConfig.from_pretrained(args.model_path)

        model = PeftModel.from_pretrained(model, 
                                                args.model_path, 
                                                config=lora_config,
                                                is_trainable=False)

        model = model.merge_and_unload()
        projector_path = os.path.join(args.model_path, "mm_projector.pth")

        if os.path.exists(projector_path):
            load_module_state(model, projector_path)
        model.to(dtype=torch.bfloat16)
        model.eval()
        model.generation_config.bos_token_id = model.config.bos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        # model.config.pad_token_id = tokenizer.pad_token_id
        # model.generation_config.bos_token_id = tokenizer.bos_token_id
        # model.config.bos_token_id = tokenizer.bos_token_id

    questions = []
    with open(args.question_file, "r") as f:
        json_data = json.load(f)
        for line in json_data:
            questions.append({"question_id": line["id"], 
                              "image": line["image"], 
                              "text": line["conversations"][0]["value"].replace("<image>\n",""),
                              "ans": line["conversations"][1]["value"]})

    # Check if answers file already exists and load existing data
    all_answers = []
    existing_question_ids = set()
    
    if os.path.exists(args.answers_file):
        with open(args.answers_file, "r") as ans_file:
            for line in ans_file:
                existing_data = json.loads(line)
                all_answers.append(existing_data)  # Store existing answers
                existing_question_ids.add(existing_data["question_id"])  # Track existing question_ids

    if args.image_aspect_ratio:
        model.config.image_aspect_ratio = args.image_aspect_ratio

    # Process new questions
    for line in tqdm(questions, total=len(questions)):
        idx = line["question_id"]
        # Skip if the answer for this question_id already exists
        if idx in existing_question_ids:
            print(f"Skipping question {idx}, already exists.")
            continue

        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        if args.lora:
            image_tensor.to(dtype=torch.bfloat16)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.to(device=device),
                images=image_tensor.unsqueeze(0).to(device=device),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        new_answer = {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }

        all_answers.append(new_answer)  # Add new answers to the list

    with open(args.answers_file, "w") as ans_file:
        for answer in all_answers:
            ans_file.write(json.dumps(answer) + "\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--lora", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="apple/FastVLM-1.5B")
    parser.add_argument("--image_aspect_ratio", type=str, default="square")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
