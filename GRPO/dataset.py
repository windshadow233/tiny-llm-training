from datasets import load_dataset
from torch.utils.data import Dataset


class GSM8kDataset(Dataset):
    def __init__(self, tokenizer, split='train', data_range=None, max_length=512):
        dataset = load_dataset("openai/gsm8k", 'main', split=split)
        if isinstance(data_range, tuple) and len(data_range) == 2:
            start, end = data_range
            self.dataset = dataset.select(range(start, end))
        else:
            self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = """You are given a problem.
Think about it and provide your working out.
Place it between <think> and </think>.
Then, provide your numeric answer between <answer> and </answer>. For example:
<think>
...
</think>
<answer>
...
</answer>
"""

    def __len__(self):
        return len(self.dataset)
    
    def extract_hash_answer(self, text: str):
        if "####" not in text:
            return None
        return text.split("####")[1].strip().replace(',', '')

    def __getitem__(self, idx):
        data = self.dataset[idx]
        question = data['question']
        answer = self.extract_hash_answer(data['answer'])
        
        prompt = self.tokenizer.apply_chat_template([
            {"role": "system", 'content': self.system_prompt},
            {"role": "user", 'content': question}
        ], add_generation_prompt=True, tokenize=False)
        
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'][0]
        inputs['attention_mask'] = inputs['attention_mask'][0]
        inputs['answer'] = str(answer)
        inputs['prompt'] = prompt
        
        return inputs
