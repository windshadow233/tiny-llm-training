import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import os
from tqdm import tqdm
from torch.optim import AdamW
from collections import defaultdict
import copy
from transformers import get_scheduler, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from GRPO.dataset import GSM8kDataset
from GRPO.cot_reward import extract_answer, accuracy_reward, strict_format_reward, soft_format_reward
from utils import load_model, color_text, center, GRPO_MODEL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Buffer(Dataset):
    def __init__(self):
        self.buffer = defaultdict(list)

    def extend(self, items):
        for key, value in items.items():
            if value is None:
                continue
            self.buffer[key].extend(value)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        if not self.buffer:
            return 0
        return len(next(iter(self.buffer.values())))

    def __getitem__(self, idx):
        item = {}
        for key, value in self.buffer.items():
            item[key] = value[idx]
        return item


class GRPOArgs:
    lr = 1e-6
    num_epochs = 1
    group_size = 16  # 组内样本数
    max_prompt_length = 256  # 最大输入长度
    max_generate_length = 256  # 最大输出长度
    temperature = 1.0
    beta = 0.0  # KL散度系数
    clip_eps = 0.2
    gradient_accumulation_steps = 4  # 梯度累加
    warmup_ratio = 0.0  # 学习率预热比例
    num_iterations = 1  # 每次采样得到的样本训练模型迭代次数
    batch_size = 4
    log_dir = 'runs/grpo'
    output_dir = './model/grpo'
    save_steps = 200  # 保存模型的步数
    max_grad_norm = 1.0


class GRPOTrainer:
    def __init__(self, model, train_dataset, tokenizer, reward_fcns, args: GRPOArgs):
        self.args = args
        assert (self.args.gradient_accumulation_steps * self.args.batch_size) % self.args.group_size == 0, "The equivalent batch size must be divisible by the group size."
        self.model = model
        self.model_ref = None
        if args.beta != 0.0:
            self.model_ref = copy.deepcopy(model).half().eval()
            self.model_ref.requires_grad_(False)
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.reward_fcns = reward_fcns
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        self.equivalent_batch_size = 0
        self.num_training_steps = 0
        self.parameters = []
        self.max_length = self.args.max_prompt_length + self.args.max_generate_length

        self.writer = SummaryWriter(self.args.log_dir)
        
        self.prepare()

    def prepare(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
        )

        self.model = get_peft_model(self.model, lora_config)

        for name, param in self.model.named_parameters():
            if 'lora' in name:
                self.parameters.append(param)

        self.optimizer = AdamW(self.parameters, lr=self.args.lr)
        
        self.equivalent_batch_size = self.args.gradient_accumulation_steps * self.args.batch_size

        num_update_steps_per_epoch = len(self.train_dataset) * self.args.group_size // self.equivalent_batch_size * self.args.num_iterations
        self.num_training_steps = num_training_steps = self.args.num_epochs * num_update_steps_per_epoch

        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps
        )
        components = [self.model, self.optimizer, self.scheduler]
        if self.model_ref is not None:
            components.append(self.model_ref)

        prepared = self.accelerator.prepare(*components)

        if self.model_ref is not None:
            (self.model,
             self.optimizer,
             self.scheduler,
             self.model_ref) = prepared
        else:
            (self.model,
             self.optimizer,
             self.scheduler) = prepared

    @staticmethod
    def action_logsoftmax(logits, chosen_ids):
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.gather(2, chosen_ids.unsqueeze(-1)).squeeze(-1)

    def generate_group(self, data):
        prompt = data['prompt']
        answer = data['answer']
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.args.max_prompt_length, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.model.device)[0]
        attention_mask = inputs['attention_mask'].to(self.model.device)[0]

        prompt_ids = torch.stack([input_ids] * self.args.group_size, dim=0)
        prompt_att_mask = torch.stack([attention_mask] * self.args.group_size, dim=0)
        prompt_length = prompt_ids.shape[1]

        generated_ids = self.model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_att_mask,
            max_new_tokens=self.args.max_generate_length,
            do_sample=True,
            temperature=self.args.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=False
        )
        generated_length = generated_ids.shape[1]
        if generated_length >= self.max_length:
            generated_ids = generated_ids[:, :self.max_length]
        else:
            padding = torch.full((self.args.group_size, self.max_length - generated_length), fill_value=self.tokenizer.pad_token_id, device=generated_ids.device)
            generated_ids = torch.cat([generated_ids, padding], dim=1)
        generated_attention_mask = generated_ids.ne(self.tokenizer.pad_token_id).long()
        response_ids = generated_ids[:, prompt_length:]
        action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).long()
        group = {
            'generated_ids': generated_ids,
            'attention_mask': generated_attention_mask,
            'response_ids': response_ids,
            'action_mask': action_mask,
            'answer': answer,
            'prompt': prompt
        }
        return group

    def compute_rewards(self, prompt, responses, answer):
        rewards = []
        for reward_fcn in self.reward_fcns:
            rewards.append(reward_fcn(prompt, responses, answer))
        return torch.tensor(rewards, dtype=torch.float32, device=self.model.device).T

    @torch.no_grad()
    def generate_group_data(self, data):
        group = self.generate_group(data)
        generated_ids = group['generated_ids']
        attention_mask = group['attention_mask']
        response_ids = group['response_ids']
        action_mask = group['action_mask']
        num_actions = action_mask.shape[1]
        answer = group['answer']
        prompt = group['prompt']

        logits = self.model(generated_ids, attention_mask=attention_mask).logits
        log_prob_old = self.action_logsoftmax(logits[:, :-1], generated_ids[:, 1:])[:, -num_actions:]

        log_prob_ref = None
        if self.model_ref is not None:
            logits_ref = self.model_ref(generated_ids, attention_mask=attention_mask).logits
            log_prob_ref = self.action_logsoftmax(logits_ref[:, :-1], generated_ids[:, 1:])[:, -num_actions:]

        response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        rewards = self.compute_rewards(prompt, response_texts, answer) # (group_size, num_rewards)

        total_rewards = rewards.sum(dim=1)
        group_reward_mean = total_rewards.mean()
        group_reward_std = total_rewards.std()

        adv = (total_rewards - group_reward_mean) / (group_reward_std + 1e-6)

        self.model.train()
        return {
            'generated_ids': generated_ids,
            'attention_mask': attention_mask,
            'action_mask': action_mask,
            'log_prob_old': log_prob_old,
            'log_prob_ref': log_prob_ref,
            'advantages': adv,
            'rewards': rewards  # (group_size, num_rewards)
        }

    def compute_loss(self, batch_data):
        generated_ids = batch_data['generated_ids']
        attention_mask = batch_data['attention_mask']
        action_mask = batch_data['action_mask']
        num_actions = action_mask.shape[1]
        log_prob_old = batch_data['log_prob_old']
        advantages = batch_data['advantages'].unsqueeze(-1)

        logits_new = self.model(generated_ids, attention_mask=attention_mask).logits
        log_prob_new = self.action_logsoftmax(logits_new[:, :-1], generated_ids[:, 1:])[:, -num_actions:]

        ratio = (log_prob_new - log_prob_old).exp()
        ratio_clip = ratio.clip(1 - self.args.clip_eps, 1 + self.args.clip_eps)
        loss = - torch.min(ratio * advantages, ratio_clip * advantages)
        if self.model_ref is not None:
            log_prob_ref = batch_data['log_prob_ref']
            log_ratio = (log_prob_ref - log_prob_new) * action_mask
            kl = log_ratio.exp() - 1 - log_ratio
            loss += self.args.beta * kl
        loss = loss.sum(1) / action_mask.sum(1)
        loss = loss.mean()

        return loss

    def train_step(self, batch_data):
        updated = False
        with self.accelerator.accumulate(self.model):
            loss = self.compute_loss(batch_data)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                updated = True
                self.accelerator.clip_grad_norm_(self.parameters, self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return updated, loss.item()
    
    def log_items(self, loss, rewards, global_steps):
        self.writer.add_scalar('GRPO/Loss', loss, global_steps)
        reward_mean = rewards.mean(dim=0)  # (num_rewards,)
        reward_std = rewards.std(dim=0)  # (num_rewards,)
        reward_sum_mean = rewards.sum(dim=1).mean()
        reward_sum_std = rewards.sum(dim=1).std()
        for i, reward_fcn in enumerate(self.reward_fcns):
            reward_name = reward_fcn.__name__
            self.writer.add_scalar(f'GRPO/Reward/{reward_name}/mean', reward_mean[i].item(), global_steps)
            self.writer.add_scalar(f'GRPO/Reward/{reward_name}/std', reward_std[i].item(), global_steps)
        self.writer.add_scalar('GRPO/Reward/Total/mean', reward_sum_mean.item(), global_steps)
        self.writer.add_scalar('GRPO/Reward/Total/std', reward_sum_std.item(), global_steps)
        self.writer.add_scalar('GRPO/LR', self.optimizer.param_groups[0]['lr'], global_steps)
        
    def test_one_sample(self, data):
        self.model.eval()
        with torch.inference_mode():
            prompt = data['prompt']
            answer = data['answer']
            inputs = self.tokenizer(prompt, padding='max_length', max_length=self.args.max_prompt_length, truncation=True, return_tensors='pt')
            input_ids = inputs['input_ids'].to(self.model.device)
            prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            attention_mask = inputs['attention_mask'].to(self.model.device)
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.args.max_generate_length,
                do_sample=True,
                temperature=self.args.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=False
            )[0][self.args.max_prompt_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            _, gen_answer = extract_answer(response)
            print(color_text("\n" + center("Prompt"), "cyan"))
            print(prompt)
            print(color_text("\n" + center("Generated Response"), "green"))
            print(response)
            print(color_text("\n" + center("Output Answer"), "blue"))
            print(gen_answer)
            print(color_text("\n" + center("Answer"), "yellow"))
            print(answer)
            print(color_text("\n" + center(""), "magenta"))

    def train(self):
        buffer = Buffer()
        global_steps = 0
        for idx in tqdm(range(self.args.num_epochs * len(self.train_dataset)), desc="Training Data Index", dynamic_ncols=True):
            self.model.eval()
            data = self.train_dataset[idx % len(self.train_dataset)]
            group_data = self.generate_group_data(data)
            buffer.extend(group_data)
            if len(buffer) == self.equivalent_batch_size:
                dataloader = DataLoader(
                    buffer,
                    batch_size=self.args.batch_size,
                    shuffle=False
                )
                self.model.train()
                for _ in range(self.args.num_iterations):
                    for batch in dataloader:
                        updated, loss = self.train_step(batch)
                        if updated:
                            global_steps += 1
                            if global_steps % 10 == 0:
                                self.log_items(loss, batch['rewards'], global_steps)
                                self.test_one_sample(data)
                            if global_steps % self.args.save_steps == 0:
                                self.save_model(global_steps)
                                print(color_text(f"\nModel saved at step {global_steps}", "green"))
                buffer.clear()
                
        self.save_model(global_steps)
        print(color_text(f"\nTraining completed. Final model saved at step {global_steps}", "green"))

    def save_model(self, step):
        path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


args = GRPOArgs()

tokenizer = AutoTokenizer.from_pretrained(GRPO_MODEL_NAME, trust_remote_code=True, padding_side='left',
                                          local_files_only=True)
dataset = GSM8kDataset(tokenizer=tokenizer, split='train', max_length=args.max_prompt_length)
model = load_model(GRPO_MODEL_NAME, torch_dtype=torch.float16)

trainer = GRPOTrainer(model, dataset, tokenizer, reward_fcns=[accuracy_reward, strict_format_reward, soft_format_reward], args=args)
trainer.train()
