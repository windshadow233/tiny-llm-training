from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

from RLHF.utils import pad_to_left
from utils import MODEL_NAME



class RLHFDataset(Dataset):
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

    def __getitem__(self, idx):
        data = self.dataset[idx]
        instruction = data['query']
        chosen = data['chosen']
        prompt = f"指令:{instruction}\n输出:"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        chosen_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        prompt_ids = prompt_ids[:self.max_length - 1] + [self.tokenizer.bos_token_id]
        chosen_ids = chosen_ids[:self.max_length]
        if len(chosen_ids) < self.max_length:
            chosen_ids += [self.tokenizer.eos_token_id]
        chosen_ids = chosen_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(chosen_ids))
        
        input_ids, attention_mask = pad_to_left(prompt_ids, self.max_length, self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "chosen": torch.tensor(chosen_ids, dtype=torch.long)
        }