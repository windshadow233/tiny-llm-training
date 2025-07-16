from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import logging
import argparse
from tqdm import tqdm
import sys

from RLHF.model import RewardModel
from RLHF.dataset import RLHFDataset
from RLHF.utils import *
from utils import color_text, load_model, center, MODEL_NAME

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="RLHF PPO Training")

    parser.add_argument("--max_length", '-l', type=int, default=256)
    parser.add_argument("--num_epochs", '-e', type=int, default=1)
    parser.add_argument("--batch_size", '-b', type=int, default=2)
    parser.add_argument("--learning_rate_actor", '-lra', type=float, default=2e-6)
    parser.add_argument("--learning_rate_critic", '-lrc', type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", '-s', type=int, default=8)
    parser.add_argument("--data_range_start", '-ds', type=int, default=0)
    parser.add_argument("--data_range_end", '-de', type=int, default=25000)
    parser.add_argument("--epsilon", '-eps', type=float, default=0.2)
    parser.add_argument("--output_dir", '-o', type=str, default='model/rlhf')

    return parser.parse_args()


def train(args):
    batch_size = args.batch_size
    max_length = args.max_length
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    dataset = RLHFDataset(tokenizer=tokenizer, data_range=(args.data_range_start, args.data_range_end), max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # writer = SummaryWriter(log_dir='runs/rlhf')

    logging.info(f"Data loaded successfully. Dataset size: {len(dataset)}")

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    model_actor = load_model('model/sft', torch_dtype='auto')
    lora_parameters = []
    for name, param in model_actor.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_parameters.append(param)

    logging.info("Actor model loaded successfully.")

    model_ref = load_model('model/sft', torch_dtype='auto')
    model_ref.eval().requires_grad_(False)

    logging.info("Reference model loaded successfully.")

    model_critic = RewardModel.from_pretrained('model/reward_model', torch_dtype='auto')

    logging.info("Critic model loaded successfully.")

    model_reward = RewardModel.from_pretrained('model/reward_model', torch_dtype='auto')
    model_reward.eval().requires_grad_(False)

    logging.info("Reward model loaded successfully.")

    num_epochs = args.num_epochs
    accumulation_steps = args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(dataset) // batch_size // accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    optimizer_actor = AdamW(lora_parameters, lr=args.learning_rate_actor, betas=(0.9, 0.95))
    scheduler_actor = get_scheduler(
        name="linear",
        optimizer=optimizer_actor,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps
    )

    optimizer_critic = AdamW(model_critic.v_head.parameters(), lr=args.learning_rate_critic, betas=(0.9, 0.95))
    scheduler_critic = get_scheduler(
        name="linear",
        optimizer=optimizer_critic,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps
    )

    (model_actor,
     model_ref,
     model_critic,
     dataloader,
     optimizer_actor,
     scheduler_actor,
     optimizer_critic,
     scheduler_critic
     ) = accelerator.prepare(
        model_actor,
        model_ref,
        model_critic,
        dataloader,
        optimizer_actor,
        scheduler_actor,
        optimizer_critic,
        scheduler_critic
    )

    model_actor.train()
    model_critic.train()
    logging.info("Starting training...")

    pad = tokenizer.pad_token_id
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    eps = args.epsilon

    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(dataloader, 1), desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True, total=len(dataloader)):
            (generated_ids,
             generated_attention_mask,
             log_prob_old,
             value_old,
             reward,
             log_prob_ref) = generate_batch_data(
                model_actor,
                model_ref,
                model_critic,
                model_reward,
                batch['input_ids'],
                batch['attention_mask'],
                pad=pad,
                eos=eos
            )
            if generated_ids is None:
                continue

            end = get_eos_position(generated_ids, eos=eos)
            for idx, e in enumerate(end):
                value_old[idx, e + 1:] = 0

            kl, reward_kl = calculate_reward_with_kl(end, log_prob_old, log_prob_ref, reward, coeff=0.2)
            td_delta = calculate_td_delta(reward_kl, value_old, 1.0, max_length)
            adv = calculate_advantage(td_delta)

            with accelerator.accumulate(model_actor), accelerator.accumulate(model_critic):

                logits_new = model_actor(generated_ids, attention_mask=generated_attention_mask).logits
                log_prob_new = calculate_action_logsoftmax(logits_new[:, :-1], generated_ids[:, 1:])

                ratio = ((log_prob_new[:, max_length - 1:] - log_prob_old[:, max_length - 1:])
                         * generated_attention_mask[:, max_length:]).exp()
                
                loss_actor_1 = adv * ratio
                loss_actor_2 = adv * torch.clip(ratio, 1 - eps, 1 + eps)
                loss_actor = -torch.min(loss_actor_1, loss_actor_2).mean()

                value_new = model_critic(generated_ids, attention_mask=generated_attention_mask)
                loss_critic_1 = (value_new[:, max_length:] - adv - value_old[:, max_length:])[:, :-1] ** 2
                clip_value_new = torch.clip(value_new, value_old - eps, value_old + eps)[:, max_length:]
                loss_critic_2 = (clip_value_new - adv - value_old[:, max_length:])[:, :-1] ** 2
                loss_critic = torch.max(loss_critic_1, loss_critic_2).mean()

                accelerator.backward(loss_actor + loss_critic)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters + [p for p in model_critic.v_head.parameters()], 1.0)
                    optimizer_actor.step()
                    optimizer_critic.step()
                    scheduler_actor.step()
                    scheduler_critic.step()
                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()
                    
            global_step = epoch * len(dataloader) + step
            writer.add_scalar('RLHF/Actor-Loss', loss_actor.item(), global_step)
            writer.add_scalar('RLHF/Critic-Loss', loss_critic.item(), global_step)
            writer.add_scalar('RLHF/Reward', reward.mean().item(), global_step)
            writer.add_scalar('RLHF/KL', -kl.mean().item(), global_step)

            if step % 10 == 0 and accelerator.is_main_process:
                print(f"Step {step}, Actor Loss: {loss_actor.item():.4f}, Critic Loss: {loss_critic.item():.4f}, Reward: {reward.mean().item():.4f}")

                input_ids = batch['input_ids'][0]
                bos_pos = (input_ids == bos).nonzero(as_tuple=True)[0].item()
                prompt = input_ids[:bos_pos + 1]
                attention_mask = batch['attention_mask'][0][:bos_pos + 1]
                chosen_ids = batch['chosen'][0]

                model_actor.eval()
                with torch.inference_mode():
                    pred = model_actor.generate(
                        input_ids=prompt[None],
                        attention_mask=attention_mask[None],
                        max_new_tokens=256,
                        pad_token_id=pad,
                        eos_token_id=eos,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        repetition_penalty=1.5,
                    )[0][bos_pos + 1:]
                    prompt_text = tokenizer.decode(prompt, skip_special_tokens=True)
                    gen_text = tokenizer.decode(pred, skip_special_tokens=True)
                    answer_text = tokenizer.decode(chosen_ids, skip_special_tokens=True)

                    print(color_text("\n" + center("Prompt"), "cyan"))
                    print(prompt_text)
                    print(color_text("\n" + center("Generated Response"), "green"))
                    print(gen_text)
                    print(color_text("\n" + center("Chosen Response"), "yellow"))
                    print(answer_text)
                    print(color_text("\n" + center(""), "magenta"))

                model_actor.train()
    
    output_dir = args.output_dir
    model_actor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info("Training completed and model saved.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
    sys.exit(0)
