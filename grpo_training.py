import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import os
from tqdm import tqdm
from torch.optim import AdamW
from collections import deque
import copy
from transformers import get_scheduler, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from GRPO.dataset import GSM8kDataset
# from GRPO.code_reward import REWARD_FCNS, run_code_from_text
from GRPO.cot_reward import REWARD_FCNS, extract_answer
from utils import load_model, color_text, center, GRPO_MODEL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GRPOArgs:
    lr = 2e-5
    num_epochs = 1
    group_size = 4  # 组内样本数
    max_prompt_length = 256  # 最大输入长度
    max_generate_length = 256  # 最大输出长度
    temperature = 1.0
    beta = 0.04  # KL散度系数
    clip_eps = 0.2
    gradient_accumulation_steps = 4  # 梯度累加
    warmup_ratio = 0.01  # 学习率预热比例
    num_iterations = 1  # 每次采样得到的样本训练模型迭代次数
    batch_size = 2
    output_dir = './model/grpo'
    save_steps = 200  # 保存模型的步数


class GRPOTrainer:
    def __init__(self, model, train_dataset, tokenizer, reward_fcns, args: GRPOArgs):
        self.model = model
        self.model_ref = None
        if args.beta != 0.0:
            self.model_ref = copy.deepcopy(model).half().eval()
            for param in self.model_ref.parameters():
                param.requires_grad = False
        self.train_dataset = train_dataset
        self.dataloader = None
        self.tokenizer = tokenizer
        self.reward_fcns = reward_fcns
        self.args = args
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        self.num_training_steps = 0
        self.current_steps = 0
        self.parameters = []
        self.max_length = self.args.max_prompt_length + self.args.max_generate_length

        self.writer = SummaryWriter('runs/grpo')

    def prepare(self):
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )

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

        num_update_steps_per_epoch = len(
            self.train_dataset) // self.args.batch_size // self.accelerator.gradient_accumulation_steps * self.args.num_iterations
        self.num_training_steps = num_training_steps = self.args.num_epochs * num_update_steps_per_epoch

        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps
        )
        components = [self.model, self.dataloader, self.optimizer, self.scheduler]
        if self.model_ref is not None:
            components.append(self.model_ref)

        prepared = self.accelerator.prepare(*components)

        if self.model_ref is not None:
            (self.model,
             self.dataloader,
             self.optimizer,
             self.scheduler,
             self.model_ref) = prepared
        else:
            (self.model,
             self.dataloader,
             self.optimizer,
             self.scheduler) = prepared

    @staticmethod
    def action_logsoftmax(logits, chosen_ids):
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.gather(2, chosen_ids.unsqueeze(-1)).squeeze(-1)

    def generate_groups(self, batch):
        groups = []
        input_ids, attention_mask, answers, prompts = batch['input_ids'], batch['attention_mask'], batch['answer'], batch['prompt']
        for input_id, att_mask, answer, prompt in zip(input_ids, attention_mask, answers, prompts):

            prompt_ids = torch.stack([input_id] * self.args.group_size, dim=0)
            prompt_att_mask = torch.stack([att_mask] * self.args.group_size, dim=0)
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
                padding = torch.full((self.args.group_size, self.max_length - generated_length),
                                     fill_value=self.tokenizer.pad_token_id, device=generated_ids.device)
                generated_ids = torch.cat([generated_ids, padding], dim=1)
            generated_attention_mask = generated_ids.ne(self.tokenizer.pad_token_id).long()
            response_ids = generated_ids[:, prompt_length:]
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).long()
            groups.append({
                'generated_ids': generated_ids,
                'attention_mask': generated_attention_mask,
                'response_ids': response_ids,
                'action_mask': action_mask,
                'num_actions': action_mask.shape[1],
                'answer': answer,
                'prompt': prompt
            })
        return groups

    def compute_rewards(self, prompt, responses, answer):
        rewards = []
        for reward_fcn in self.reward_fcns.values():
            rewards.append(reward_fcn(prompt, responses, answer))
        return torch.tensor(rewards, dtype=torch.float32, device=self.model.device).clamp(-5, 5)

    @torch.no_grad()
    def generate_batch_data(self, batch):
        self.model.eval()
        groups = self.generate_groups(batch)
        generated_ids_list = []
        attention_mask_list = []
        action_mask_list = []
        log_prob_old_list = []
        log_prob_ref_list = []
        advantage_list = []
        rewards_list = []
        for group in groups:
            generated_ids = group['generated_ids']
            attention_mask = group['attention_mask']
            response_ids = group['response_ids']
            action_mask = group['action_mask']
            num_actions = group['num_actions']
            answer = group['answer']
            prompt = group['prompt']

            generated_ids_list.append(generated_ids)
            attention_mask_list.append(attention_mask)
            action_mask_list.append(action_mask)

            logits = self.model(generated_ids, attention_mask=attention_mask).logits
            log_prob_old = self.action_logsoftmax(logits[:, :-1], generated_ids[:, 1:])[:, -num_actions:]
            log_prob_old_list.append(log_prob_old)

            if self.model_ref is not None:
                logits_ref = self.model_ref(generated_ids, attention_mask=attention_mask).logits
                log_prob_ref = self.action_logsoftmax(logits_ref[:, :-1], generated_ids[:, 1:])[:, -num_actions:]
                log_prob_ref_list.append(log_prob_ref)

            response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            rewards = self.compute_rewards(prompt, response_texts, answer)

            total_rewards = rewards.sum(dim=0)
            group_reward_mean = total_rewards.mean()
            group_reward_std = total_rewards.std()

            adv = (total_rewards - group_reward_mean) / (group_reward_std + 1e-6)
            advantage_list.append(adv)
            rewards_list.append(rewards)

        self.model.train()
        return {
            'generated_ids': torch.cat(generated_ids_list, dim=0),
            'attention_mask': torch.cat(attention_mask_list, dim=0),
            'action_mask': torch.cat(action_mask_list, dim=0),
            'log_prob_old': torch.cat(log_prob_old_list, dim=0),
            'log_prob_ref': torch.cat(log_prob_ref_list, dim=0) if self.model_ref is not None else None,
            'advantages': torch.cat(advantage_list, dim=0),
            'rewards': torch.stack(rewards_list, dim=0)  # (B, num_rewards, group_size)
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
        with self.accelerator.accumulate(self.model):
            self.current_steps += 1
            loss = self.compute_loss(batch_data)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.parameters, 0.1)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                ...
                self.writer.add_scalar('GRPO/Loss', loss.item(), self.current_steps)
        rewards = batch_data['rewards']  # (B, num_rewards, group_size)
        reward_mean = rewards.mean(dim=(0, 2))  # (num_rewards,)
        reward_sum_mean = rewards.sum(dim=1).mean()
        for i, reward_name in enumerate(self.reward_fcns.keys()):
            self.writer.add_scalar(f'GRPO/Reward/{reward_name}', reward_mean[i].item(), self.current_steps)
        self.writer.add_scalar('GRPO/Reward/Total', reward_sum_mean.item(), self.current_steps)
        torch.cuda.empty_cache()

    def train(self):
        buffer = deque(maxlen=self.args.gradient_accumulation_steps)
        for epoch in range(self.args.num_epochs):
            for step, batch in tqdm(enumerate(self.dataloader, 1), desc=f"Epoch {epoch + 1}/{self.args.num_epochs}",
                                    dynamic_ncols=True, total=len(self.dataloader)):
                batch_data = self.generate_batch_data(batch)
                buffer.append(batch_data)
                if step % self.args.gradient_accumulation_steps == 0:
                    self.model.train()
                    for _ in range(self.args.num_iterations):
                        for batch_data in buffer:
                            self.train_step(batch_data)
                if step % 10 == 0 and self.accelerator.is_main_process:
                    self.model.eval()
                    with torch.inference_mode():
                        input_ids = batch['input_ids'][0]
                        attention_mask = batch['attention_mask'][0]
                        answer = batch['answer'][0]
                        generated_ids = self.model.generate(
                            input_ids=input_ids[None],
                            attention_mask=attention_mask[None],
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=self.args.temperature,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            early_stopping=False
                        )[0][self.args.max_prompt_length:]
                        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                        # gen_answer = run_code_from_text(response)
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
                if step % self.args.save_steps == 0:
                    self.save_model()
                    print(color_text(f"\nModel saved at step {step}", "green"))

    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)


args = GRPOArgs()

tokenizer = AutoTokenizer.from_pretrained(GRPO_MODEL_NAME, trust_remote_code=True, padding_side='left',
                                          local_files_only=True)
dataset = GSM8kDataset(tokenizer=tokenizer, split='train', max_length=args.max_prompt_length)
model = load_model(GRPO_MODEL_NAME, torch_dtype=torch.float16)

trainer = GRPOTrainer(model, dataset, tokenizer, reward_fcns=REWARD_FCNS, args=args)
trainer.prepare()
trainer.train()
