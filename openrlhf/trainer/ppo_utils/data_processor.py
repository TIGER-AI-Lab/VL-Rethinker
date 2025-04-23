# /*
#  * Modified by Haozhe Wang in 2025
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  */
import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor
from transformers.processing_utils import ProcessorMixin
try:
    from transformers import Qwen2_5_VLProcessor
except Exception as e:
    print("Qocal Qwen2_5_VLProcessor not found")

# https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_5_vl.md
class BaseDataProcessor(ABC):
    def __init__(self, processor: ProcessorMixin):
        super().__init__()
        self.processor = processor

    @abstractmethod
    def __call__(
        self,
        messages: Union[Dict, List[str], str],
        max_length: int,
        padding: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: Optional[bool] = False,
        truncation: Optional[bool] = True,
    ) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def make_input_batch(self, inputs: List[Dict]) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def split_input_batch(self, batch: Dict) -> List[Dict]:
        raise NotImplementedError

    def _format_messages(self, messages: Union[Dict, List[str], str]) -> List[Dict]:
        if isinstance(messages, list) and isinstance(messages[0], str):
            return [json.loads(m) for m in messages]
        elif isinstance(messages, str):
            return [json.loads(messages)]
        elif isinstance(messages, dict):
            return [messages]
        else:
            raise ValueError("Invalid messages format, must be a list of strings or a string or a dict")

    def apply_chat_template(
        self,
        messages: Union[Dict, List[str], str],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> List[str]:
        messages = self._format_messages(messages)
        
        return self.processor.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )

    def get_images_from_messages(
        self, messages: Union[Dict, List[str], str]
    ) -> List[Dict]:
        messages = self._format_messages(messages)
        return self._get_images_from_messages(messages)

    @abstractmethod
    def _get_images_from_messages(self, messages: List[Dict]) -> List[Dict]:
        raise NotImplementedError

    @property
    def pad_token_id(self) -> int:
        return self.processor.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.processor.tokenizer.eos_token_id

    @property
    def tokenizer(self):
        return self.processor.tokenizer


def add_pixel_bounds(messages):
    # 默认的像素范围
    DEFAULT_MIN_PIXELS = int(os.getenv("MIN_PIXELS", 256 * 28 * 28))
    DEFAULT_MAX_PIXELS = int(os.getenv("MAX_PIXELS", 1280 * 28 * 28))

    def process_content(content):
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    if "min_pixels" not in item:
                        item["min_pixels"] = DEFAULT_MIN_PIXELS
                    if "max_pixels" not in item:
                        item["max_pixels"] = DEFAULT_MAX_PIXELS
        return content

    for message in messages:
        for msg in message:
            msg["content"] = process_content(msg["content"])
    return messages

def remove_except_last(text, tag):
    cnt = text.count(tag)
    if cnt>1: 
        index = text.rfind(tag)
        return text[:index].replace(tag, "")+text[index:]
    else: return text 
    
def find_rank_occurrence(ids, target, rank):
    """
    Finds the position (index) of the rank-th occurrence of the target in the list ids.
    
    Args:
        ids (list): List of integers to search through.
        target (int): Integer to find.
        rank (int): The occurrence number to locate (1-based).
    
    Returns:
        int: Index of the rank-th occurrence, or -1 if it doesn’t exist.
    """
    count = 0
    for i, val in enumerate(ids):
        if val == target:
            count += 1
            if count == rank:
                return i
    return -1
    
class Qwen2VLDataProcessor(BaseDataProcessor):
    def __call__(
        self,
        messages,
        max_length,
        padding=True,
        device=None,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
    ) -> Dict:
                
        # messages = newlist
        messages = self._format_messages(messages) # list of dicts
        processor = self.processor
        # for entry in messages:
        #     if entry['role'] == 'user':
        #         content = entry['content'][-1]['text']
        #         if "<image>" in content:
        #             content = content.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        #         entry['content'][-1]['text'] = content 
        
        texts = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts = self.handle_placeholders(texts)
        messages = add_pixel_bounds(messages)
        image_inputs, video_inputs = process_vision_info(messages)
        # print(texts)
        max_length = 10240 # we need to make sure it does not trucate
        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=padding,
            max_length=max_length,
            add_special_tokens=False,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        if device:
            return {k: v.to(device) for k, v in batch.items()}
        return {k: v for k, v in batch.items()}

    def handle_placeholders(self, texts):
        newlist = []
        placeholder = "<image>"
        # placeholder2 = "<image1>"
        replacewith = "<|vision_start|><|image_pad|><|vision_end|>"
        for m in texts:
            new = m 
            for k in ["<|vision_start|>","<|image_pad|>","<|vision_end|>"]:
                new = new.replace(k,"")
            # now new has no replacewith 
            if new.count(placeholder)>0:
                new = new.replace(placeholder, replacewith)
            else: 
                new = replacewith + new
            newlist.append(new)
        return newlist
        
    def make_input_batch(self, inputs: List[Dict]) -> Dict:
        # each element has no batch dimension
        batch = {k: None for k in inputs[0].keys()}
        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in inputs], dim=0)
            elif k in ["pixel_values", "image_grid_thw"]:
                # qwen2vl concat all patches of all images in a batch in the first dimension
                batch[k] = torch.cat([inp[k] for inp in inputs], dim=0)
            else:
                raise ValueError(f"Unknown key {k} for Qwen2VLDataProcessor")
        return batch

    def split_input_batch(self, batch: Dict) -> List[Dict]:
        batch_size = len(batch["input_ids"])
        batch_kwargs = [{} for _ in range(batch_size)]
        # first process None values
        keys = []
        for k, v in batch.items():
            if v is not None:
                keys.append(k)
            else:
                for i in range(batch_size):
                    batch_kwargs[i][k] = None

        if "pixel_values" in keys and (
            "input_ids" not in keys or "image_grid_thw" not in keys
        ):
            raise ValueError(
                "Cannot split batch with pixel_values without input_ids and image_grid_thw"
            )
        if "image_grid_thw" in keys and ("input_ids" not in keys):
            raise ValueError("Cannot split batch with image_grid_thw without input_ids")
        for k in ["input_ids", "attention_mask"]:
            if k in keys:
                vals = batch[k]
                if isinstance(vals, torch.Tensor):
                    vals = torch.unbind(vals)
                assert batch_size == len(vals)
                for i, v in enumerate(vals):
                    batch_kwargs[i][k] = v
        if "pixel_values" in keys:
            thws = batch["image_grid_thw"]  # (total_img_num, (t,h,w))
            pixel_values = batch["pixel_values"]
            vision_start_id = self.processor.tokenizer("<|vision_start|>")["input_ids"][0]
            vision_end_id = self.processor.tokenizer("<|vision_end|>")["input_ids"][0]
            img_idx = 0
            patch_idx = 0
            for i in range(batch_size):
                input_ids_i = batch_kwargs[i]["input_ids"]
                if not isinstance(input_ids_i, torch.Tensor):
                    input_ids_i = torch.tensor(input_ids_i)
                vision_start_num = (input_ids_i == vision_start_id).sum().item()
                vision_end_num = (input_ids_i == vision_end_id).sum().item()
                    
                img_num = vision_end_num
                if img_num == 0:
                    batch_kwargs[i]["pixel_values"] = None
                    batch_kwargs[i]["image_grid_thw"] = None
                    continue
                thws_i = thws[img_idx:img_num+img_idx]
                img_idx += img_num 
                flag = False 
                if len(thws_i) != img_num:
                    thws_i = thws[-img_num:]
                    print(f'[warning] the image_grid_thw does not match, this is polluted data, attempting: {len(thws_i)} vs {img_num}')
                    flag = True 
                # thws = thws[img_num:]
                if not isinstance(thws_i, torch.Tensor):
                    thws_i = torch.stack(thws_i)
                batch_kwargs[i]["image_grid_thw"] = thws_i
                patchs_num = thws_i.prod(dim=1).sum().item()
                pixel_values_i = pixel_values[patch_idx:patchs_num+patch_idx]
                if len(pixel_values_i) != patchs_num:
                    pixel_values_i = pixel_values[-patchs_num:]
                    print(f'[warning] the pixel_values_i does not match, this is polluted data, attempting: {patchs_num} in {len(pixel_values)} resulting in {len(pixel_values_i)}')
                    flag = True 
                # assert len(pixel_values_i) == patchs_num
                # pixel_values = pixel_values[patch_idx:patchs_num+patch_idx]
                batch_kwargs[i]["pixel_values"] = pixel_values_i
                if flag:
                    batch_kwargs[i] = None 
                    print('[truncation warning] appears a sample has mismatched vision_start and vision_end, likely due to garbage outputs, its current length is ', len(input_ids_i))
                    # print(input_ids_i.detach().cpu().numpy().tolist())
                    error_index  = find_rank_occurrence(input_ids_i.detach().cpu().numpy().tolist(), vision_start_id, 1)
                    input_ids_i[error_index:] = self.eos_token_id # how about directly before the vision start?
                    continue 
            # assert len(thws) == 0
            # assert len(pixel_values) == 0
        return batch_kwargs

    def _get_images_from_messages(self, messages: List[Dict]) -> List[Dict]:
        messages = add_pixel_bounds(messages)
        image_inputs, _ = process_vision_info(messages)
        return image_inputs


DATA_PROCESSOR_MAP = {
    Qwen2VLProcessor: Qwen2VLDataProcessor,
    Qwen2_5_VLProcessor: Qwen2VLDataProcessor,   
}
