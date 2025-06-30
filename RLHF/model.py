from transformers import AutoModelForCausalLM
from torch import nn
import torch
import os
from utils import MODEL_NAME


class RewardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype='auto'
        )
        self.base_model.requires_grad_(False)
        self.v_head = nn.Linear(self.base_model.config.hidden_size, 1, bias=False)
        self.eos_token_id = self.base_model.config.eos_token_id
        self.v_head_dtype = self.v_head.weight.dtype
        
    @property
    def device(self):
        return self.base_model.device
        
    def forward(self, input_ids, attention_mask):
        hidden_states = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].to(self.v_head_dtype)
        v = self.v_head(hidden_states).squeeze(-1)
        return v

    def save_pretrained(self, output_dir):
        self.base_model.save_pretrained(output_dir)
        torch.save(self.v_head.state_dict(), os.path.join(output_dir, 'v_head.pt'))

    @classmethod
    def from_pretrained(cls, model_path):
        model = cls()
        model.v_head.load_state_dict(torch.load(os.path.join(model_path, 'v_head.pt')))
        model.base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype='auto')
        return model

    @torch.no_grad()
    def get_reward(self, input_ids, attention_mask=None, values=None):
        if values is None:
            assert attention_mask is not None, "attention_mask must be provided if values are not given"
            values = self(input_ids, attention_mask)
        rewards = []
        for input_id, value in zip(input_ids, values):
            end = -1
            if self.eos_token_id in input_id:
                end = input_id.tolist().index(self.eos_token_id)
            rewards.append(value[end])
        return torch.stack(rewards)
