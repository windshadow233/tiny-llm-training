from transformers import get_scheduler, AutoTokenizer
import torch
from torch.nn.functional import logsigmoid
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import logging
import argparse
from tqdm import tqdm
import sys

from DPO.dataset import DPODataset, collate_fn, masked_sum
from RLHF.utils import calculate_action_logsoftmax
from utils import color_text, center, load_model, MODEL_NAME

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training")

    parser.add_argument("--max_length", '-l', type=int, default=256)
    parser.add_argument("--num_epochs", '-e', type=int, default=1)
    parser.add_argument("--batch_size", '-b', type=int, default=2)
    parser.add_argument("--learning_rate", '-lr', type=float, default=1e-6)
    parser.add_argument("--gradient_accumulation_steps", '-s', type=int, default=8)
    parser.add_argument("--data_range_start", '-ds', type=int, default=0)
    parser.add_argument("--data_range_end", '-de', type=int, default=25000)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--output_dir", '-o', type=str, default='model/dpo')

    return parser.parse_args()


def train(args):
    batch_size = args.batch_size
    max_length = args.max_length
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    dataset = DPODataset(tokenizer=tokenizer, data_range=(args.data_range_start, args.data_range_end), max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    writer = SummaryWriter(log_dir='runs/dpo')

    logging.info(f"Data loaded successfully. Dataset size: {len(dataset)}")

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    model = load_model('model/sft', torch_dtype='auto')
    lora_parameters = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.data = param.data.float()
            param.requires_grad = True
            lora_parameters.append(param)
        else:
            param.requires_grad = False

    logging.info("Actor model loaded successfully.")

    model_ref = load_model('model/sft', torch_dtype='auto')
    model_ref.eval().requires_grad_(False)

    logging.info("Reference model loaded successfully.")

    num_epochs = args.num_epochs
    accumulation_steps = args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(dataset) // batch_size // accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    optimizer = AdamW(lora_parameters, lr=args.learning_rate, betas=(0.9, 0.95))
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps
    )
    (model,
     model_ref,
     dataloader,
     optimizer,
     scheduler
     ) = accelerator.prepare(
        model,
        model_ref,
        dataloader,
        optimizer,
        scheduler
    )
    bos = tokenizer.bos_token_id
    model.train()
    beta = args.beta
    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(dataloader, 1), desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True, total=len(dataloader)):
            input_ids = batch['input_ids']
            label_mask = batch.pop('label_mask')
            with torch.no_grad():
                logits_ref = model_ref(**batch).logits
                log_prob_ref = calculate_action_logsoftmax(logits_ref[:, :-1], input_ids[:, 1:])
                log_prob_ref = masked_sum(log_prob_ref, label_mask)
                log_prob_ref_chosen, log_prob_ref_rejected = log_prob_ref.chunk(2, dim=0)
            with accelerator.accumulate(model):
                logits = model(**batch).logits
                log_prob = calculate_action_logsoftmax(logits[:, :-1], input_ids[:, 1:])
                log_prob = masked_sum(log_prob, label_mask)
                log_prob_chosen, log_prob_rejected = log_prob.chunk(2, dim=0)

                reward_chosen = log_prob_chosen - log_prob_ref_chosen
                reward_rejected = log_prob_rejected - log_prob_ref_rejected
                loss = -logsigmoid(beta * (reward_chosen - reward_rejected)).mean()
                reward_chosen = reward_chosen.detach().mean()
                reward_rejected = reward_rejected.detach().mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            global_step = epoch * len(dataloader) + step
            writer.add_scalar('DPO/Loss', loss.item(), global_step)
            writer.add_scalar('DPO/Chosen-Reward', reward_chosen.item(), global_step)
            writer.add_scalar('DPO/Rejected-Reward', reward_rejected.item(), global_step)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
                input_ids = input_ids[0]
                bos_pos = (input_ids == bos).nonzero(as_tuple=True)[0].item()
                prompt = input_ids[:bos_pos + 1]
                attention_mask = batch['attention_mask'][0][:bos_pos + 1]
                model.eval()
                with torch.inference_mode():
                    pred = model.generate(
                        input_ids=prompt[None],
                        attention_mask=attention_mask[None],
                        max_new_tokens=256,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        repetition_penalty=1.5,
                    )[0][bos_pos + 1:]
                    prompt_text = tokenizer.decode(prompt, skip_special_tokens=True)
                    gen_text = tokenizer.decode(pred, skip_special_tokens=True)
                    answer_text = tokenizer.decode(input_ids[bos_pos + 1:], skip_special_tokens=True)
                    print(color_text("\n" + center("Prompt"), "cyan"))
                    print(prompt_text)
                    print(color_text("\n" + center("Generated Response"), "green"))
                    print(gen_text)
                    print(color_text("\n" + center("Chosen Response"), "yellow"))
                    print(answer_text)
                    print(color_text("\n" + center(""), "magenta"))
                    
                model.train()

    output_dir = args.output_dir
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info("Training completed and model saved.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
    sys.exit(0)