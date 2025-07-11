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
        batchsize = len(input_ids) // 2
        
        total_loss = 0.0
        value_chosen = 0.0
        value_rejected = 0.0

        for chosen, rejected, v_chosen, v_rejected in zip(
            input_ids[:batchsize],
            input_ids[batchsize:],
            v[:batchsize],
            v[batchsize:]
        ):
            diff_start = (chosen != rejected).nonzero(as_tuple=True)[0][0].item()
            eos_chosen = (chosen == self.eos_token_id).nonzero(as_tuple=True)[0]
            end_chosen = eos_chosen[0].item() if eos_chosen.numel() > 0 else len(chosen) - 1

            eos_reject = (rejected == self.eos_token_id).nonzero(as_tuple=True)[0]
            end_reject = eos_reject[0].item() if eos_reject.numel() > 0 else len(rejected) - 1
            
            end = max(end_chosen, end_reject)
            v_chosen = v_chosen[diff_start:end + 1]
            v_rejected = v_rejected[diff_start:end + 1]
            
            loss = -nn.functional.logsigmoid(v_chosen - v_rejected).mean()
            
            total_loss += loss
            value_chosen += v_chosen.mean().item()
            value_rejected += v_rejected.mean().item()
        
        return total_loss / batchsize, value_chosen / batchsize, value_rejected / batchsize

    def save_pretrained(self, output_dir):
        self.base_model.half().save_pretrained(output_dir, safe_serialization=True)
        torch.save(self.v_head.state_dict(), os.path.join(output_dir, 'v_head.pt'))

    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=None):
        model = cls()
        model.v_head.load_state_dict(torch.load(os.path.join(model_path, 'v_head.pt')))
        model.base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch_dtype)
        return model
