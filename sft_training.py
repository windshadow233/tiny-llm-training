from transformers import AutoModelForCausalLM, get_scheduler
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
import logging
import argparse
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from SFT.dataset import SFTDataset
from utils import color_text

logging.basicConfig(level=logging.INFO)


def count_params(model):
    count_all = [i.numel() for i in model.parameters()]
    count_all = sum(count_all) / 1_000_000_000

    count_require = [i.numel() for i in model.parameters() if i.requires_grad]
    count_require = sum(count_require) / 1000_000_000

    ratio = count_require / count_all

    return {
        'count_require': count_require,
        'count_all': count_all,
        'ratio': ratio
    }


def parse_args():
    parser = argparse.ArgumentParser(description="SFT with LoRA")

    parser.add_argument("--max_length", '-l', type=int, default=256)
    parser.add_argument("--output_dir", '-o', type=str, default="model/actor")

    parser.add_argument("--batch_size", '-b', type=int, default=2)
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
    dataset = SFTDataset('hfl/alpaca_zh_51k', split='train', max_length=max_length)
    tokenizer = dataset.tokenizer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    logging.info(f"Data loaded successfully. Dataset size: {len(dataset)}")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained('m-a-p/CT-LLM-Base', trust_remote_code=True)

    logging.info("Model loaded successfully.")

    r = args.local_r
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
        if "lora" not in name:
            param.requires_grad = False
        elif param.requires_grad:
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
        for step, batch in tqdm(enumerate(dataloader, 1), desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(dataloader)):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
                input_ids = batch['input_ids'][0]
                bos_pos = (input_ids == tokenizer.bos_token_id).nonzero(as_tuple=True)[0].item()
                prompt = input_ids[:bos_pos + 1]
                attention_mask = batch['attention_mask'][0][:bos_pos + 1]
                with torch.no_grad():
                    model.eval()
                    pred = model.generate(
                        input_ids=prompt[None],
                        attention_mask=attention_mask[None],
                        max_new_tokens=256,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        repetition_penalty=1.2,
                    )[0][bos_pos + 1:]
                    prompt_text = tokenizer.decode(prompt, skip_special_tokens=True)
                    gen_text = tokenizer.decode(pred, skip_special_tokens=True)
                    answer_text = tokenizer.decode(input_ids[bos_pos + 1:], skip_special_tokens=True)
                    print(color_text("\n=== Prompt ===", "cyan"))
                    print(prompt_text)

                    print(color_text("\n=== Generated Response ===", "green"))
                    print(gen_text)

                    print(color_text("\n=== Ground Truth Answer ===", "yellow"))
                    print(answer_text)

                    print(color_text("=" * 80, "magenta"))
                    model.train()

    output_dir = args.output_dir
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
    logging.info("Training completed and model saved.")