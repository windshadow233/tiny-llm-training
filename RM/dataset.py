from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from utils import MODEL_NAME


class RMDataset(Dataset):
    def __init__(self, split='train', data_range=None, max_length=512):
        dataset = load_dataset('OpenLLMAI/comparison_data', split=split)
        if isinstance(data_range, tuple) and len(data_range) == 2:
            start, end = data_range
            self.dataset = dataset.select(range(start, end))
        else:
            self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def build_inputs(self, prompt_ids, response_ids):
        input_ids = prompt_ids + [self.tokenizer.bos_token_id] + response_ids
        input_ids = input_ids[:self.max_length]
        if len(input_ids) < self.max_length:
            input_ids += [self.tokenizer.eos_token_id]
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len
        return input_ids, attention_mask

    def __getitem__(self, idx):
        data = self.dataset[idx]
        instruction = data['query']
        chosen = data['chosen']
        rejected = data['rejected']
        prompt = f"指令:{instruction}\n输出:"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        chosen_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(rejected, add_special_tokens=False)

        chosen_input_ids, chosen_attention_mask = self.build_inputs(prompt_ids, chosen_ids)
        rejected_input_ids, rejected_attention_mask = self.build_inputs(prompt_ids, rejected_ids)

        return {
            "chosen": torch.tensor(chosen_input_ids, dtype=torch.long),
            "rejected": torch.tensor(rejected_input_ids, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_attention_mask, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_attention_mask, dtype=torch.long)
        }


def collate_fn(batch):
    chosen = [item['chosen'] for item in batch]
    rejected = [item['rejected'] for item in batch]
    chosen_attention_mask = [item['chosen_attention_mask'] for item in batch]
    rejected_attention_mask = [item['rejected_attention_mask'] for item in batch]
    input_ids = torch.stack(chosen + rejected, dim=0)
    attention_mask = torch.stack(chosen_attention_mask + rejected_attention_mask, dim=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }