from preprocess import preprocess, preprocess_multimodal
from PIL import Image
from llava.mm_utils import process_anyres_image
import transformers
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import json
import copy
from torch.utils.data import Dataset
from arguments import DataArguments
import torch
from llava.constants import IGNORE_INDEX
import os

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 teacher_tokenizer: transformers.PreTrainedTokenizer, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 model_config):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.teacher_tokenizer = teacher_tokenizer
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.model_config = model_config

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    def process_image(self, image_path, processor, aspect_ratio, image_grid_pinpoints):
        image = Image.open(image_path).convert('RGB')
        image_size = image.size
        
        if aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height: return pil_img
                res = Image.new(pil_img.mode, (max(width, height), max(width, height)), background_color)
                res.paste(pil_img, ((max(width, height) - width) // 2, (max(width, height) - height) // 2))
                return res
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        elif aspect_ratio == 'anyres':
            image = process_anyres_image(image, processor, image_grid_pinpoints)
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        return image, image_size
        

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        data_dict = {}

        if 'image' in sources[0]:
            image_path = os.path.join(self.data_args.image_folder, sources[0]['image'])
            student_img, img_size = self.process_image(image_path, 
                                                        self.data_args.image_processor, 
                                                        self.data_args.image_aspect_ratio,
                                                        self.data_args.image_grid_pinpoints)
            data_dict['image'] = student_img
            data_dict['image_size'] = img_size
            
            teacher_img, teacher_image_size = self.process_image(image_path, 
                                                self.data_args.teacher_image_processor, 
                                                self.data_args.teacher_image_aspect_ratio,
                                                self.model_config.image_grid_pinpoints)
            data_dict['teacher_image'] = teacher_img
            data_dict['teacher_image_size'] = teacher_image_size

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        teacher_sources = copy.deepcopy(sources)

        student_data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))

        teacher_data_dict = preprocess(
            teacher_sources,
            self.teacher_tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        
        data_dict['input_ids'] = student_data_dict["input_ids"][0]
        data_dict['labels'] = student_data_dict["labels"][0]
        data_dict['teacher_input_ids'] = teacher_data_dict["input_ids"][0]
        data_dict['teacher_labels'] = teacher_data_dict["labels"][0]

        if 'image' not in self.list_data_dict[i] and self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['teacher_image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    teacher_tokenizer: transformers.PreTrainedTokenizer

    def process_batch(self, 
                      instances, 
                      tokenizer, 
                      input_id_name = "input_ids", 
                      label_name = "labels",
                      image_name = 'image'):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in (input_id_name, label_name))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :tokenizer.model_max_length]
        labels = labels[:, :tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

        if image_name in instances[0]:
            images = [instance[image_name] for instance in instances]
            image_sizes = [instance[image_name + "_size"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_sizes'] = image_sizes

        return batch
    

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        student_batch = self.process_batch(instances, 
                                           self.tokenizer)
        teacher_batch = self.process_batch(instances, 
                                           self.teacher_tokenizer,
                                           input_id_name="teacher_input_ids",
                                           label_name="teacher_labels",
                                           image_name="teacher_image")

        return {
            "student": student_batch,
            "teacher": teacher_batch
        }


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                teacher_tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                model_config) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          teacher_tokenizer=teacher_tokenizer,
                                        data_path=data_args.data_path,
                                        data_args=data_args,
                                        model_config=model_config)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, teacher_tokenizer=teacher_tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)