from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, dataset_name, split='train', max_length=512):
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = AutoTokenizer.from_pretrained('m-a-p/CT-LLM-Base', trust_remote_code=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        instruction = data['instruction']
        input = data['input']
        output = data['output']
        if input:
            prompt = f"指令: {instruction}\n输入: {input}\n输出: "
        else:
            prompt = f"指令: {instruction}\n输出: "
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(output, add_special_tokens=False)

        input_ids = prompt_ids + [self.tokenizer.bos_token_id] + response_ids

        labels = [-100] * (len(prompt_ids) + 1) + response_ids + [self.tokenizer.eos_token_id]

        input_ids = input_ids[:self.max_length - 1] + [self.tokenizer.eos_token_id]
        labels = labels[:self.max_length]

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len

        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }