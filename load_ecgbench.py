from datasets import load_dataset
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGE_SAVE_DIR = "./data/ECGBench/images"
JSON_SAVE_DIR = "./data/ECGBench"

# Create a list of dataset subsets to process
subset_names = ['arena', 'code15-test', 'cpsc-test', 'csn-test-no-cot', 'ecgqa-test', 'g12-test-no-cot', 'mmmu-ecg', 'ptb-test', 'ptb-test-report']

for name in subset_names:
    dataset = load_dataset("PULSE-ECG/ECGBench", name=name, streaming=False)
    
    dataset_items = []

    def process_and_save(idx):
        item = dataset['test'][idx]
        
        image_path = item["image_path"]
        image = item["image"]
        conversations = item["conversations"]

        dataset_items.append({
            "id": item["id"],
            "image": image_path,
            "conversations": conversations
        })

        save_path = os.path.join(IMAGE_SAVE_DIR, image_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_and_save, idx) for idx in range(len(dataset['test']))]

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

    json_filename = os.path.join(JSON_SAVE_DIR, f"{name}.json")
    with open(json_filename, "w", encoding='utf-8') as json_file:
        json.dump(dataset_items, json_file, indent=4, ensure_ascii=False)

    print(f"Dataset '{name}' has been processed and saved to {json_filename}.")