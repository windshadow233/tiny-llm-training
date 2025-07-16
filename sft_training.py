from transformers import get_scheduler, AutoTokenizer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import logging
import argparse
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import sys

from SFT.dataset import SFTDataset
from utils import color_text, count_params, load_model, center, MODEL_NAME

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT with LoRA")

    parser.add_argument("--max_length", '-l', type=int, default=256)
    parser.add_argument("--output_dir", '-o', type=str, default="model/sft")

    parser.add_argument("--batch_size", '-b', type=int, default=8)
    parser.add_argument("--learning_rate", '-lr', type=float, default=1e-4)
    parser.add_argument("--num_epochs", '-e', type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", '-s', type=int, default=32)

    parser.add_argument("--lora_r", '-r', type=int, default=32)
    parser.add_argument("--lora_alpha", '-a', type=int, default=64)
    parser.add_argument("--lora_dropout", '-d', type=float, default=0.1)

    return parser.parse_args()


def train(args):
    max_length = args.max_length
    batch_size = args.batch_size

    # 读取数据
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    dataset = SFTDataset(tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter(log_dir='runs/sft')

    logging.info(f"Data loaded successfully. Dataset size: {len(dataset)}")

    # 加载模型
    model = load_model(MODEL_NAME, torch_dtype='auto')

    logging.info("Model loaded successfully.")

    r = args.lora_r
    alpha = args.lora_alpha
    dropout = args.lora_dropout
    # 使用LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )

    model = get_peft_model(model, lora_config)
    lora_parameters = []

    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_parameters.append(param)

    params_count = count_params(model)
    logging.info(f"LoRA parameters count: {params_count['count_require']} billion, "
                 f"Total parameters count: {params_count['count_all']} billion, "
                 f"Trainable parameters ratio: {params_count['ratio']:.2%}")

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    optimizer = AdamW(lora_parameters, lr=args.learning_rate)

    num_epochs = args.num_epochs
    train_batch_size = batch_size
    accumulation_steps = accelerator.gradient_accumulation_steps
    num_update_steps_per_epoch = len(dataset) // train_batch_size // accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps
    )

    model, dataloader, optimizer, scheduler = accelerator.prepare(model, dataloader, optimizer, scheduler)

    model.train()

    logging.info("Starting training...")

    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(dataloader, 1), dynamic_ncols=True, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(dataloader)):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            global_step = epoch * len(dataloader) + step
            writer.add_scalar('SFT/Loss', loss.item(), global_step)

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
                input_ids = batch['input_ids'][0]
                bos_pos = (input_ids == tokenizer.bos_token_id).nonzero(as_tuple=True)[0].item()
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
                    print(color_text("\n" + center("Ground Truth Answer"), "yellow"))
                    print(answer_text)
                    print(color_text("\n" + center(""), "magenta"))
                model.train()

    output_dir = args.output_dir
    model.half().save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    logging.info("Training completed and model saved.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
    sys.exit(0)
