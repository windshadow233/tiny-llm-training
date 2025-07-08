from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import logging
import argparse
from tqdm import tqdm
import sys

from RM.dataset import RMDataset, collate_fn
from RM.model import RewardModel
from utils import color_text, center, MODEL_NAME

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Reward Model Training")

    parser.add_argument("--max_length", '-l', type=int, default=256)
    parser.add_argument("--output_dir", '-o', type=str, default='model/reward_model')

    parser.add_argument("--batch_size", '-b', type=int, default=32)
    parser.add_argument("--learning_rate", '-lr', type=float, default=5e-5)
    parser.add_argument("--num_epochs", '-e', type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", '-s', type=int, default=8)

    parser.add_argument("--data_range_start", '-ds', type=int, default=0)
    parser.add_argument("--data_range_end", '-de', type=int, default=75000)

    return parser.parse_args()


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    dataset = RMDataset(tokenizer=tokenizer, data_range=(args.data_range_start, args.data_range_end), max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    writer = SummaryWriter(log_dir='runs/reward_model')

    logging.info(f"Data loaded successfully. Dataset size: {len(dataset)}")

    model = RewardModel()
    logging.info("Model loaded successfully.")

    optimizer = AdamW(model.v_head.parameters(), lr=args.learning_rate)
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    accumulation_steps = args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(dataset) // args.batch_size // accumulation_steps
    num_training_steps = args.num_epochs * num_update_steps_per_epoch

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps
    )

    model, dataloader, optimizer, scheduler = accelerator.prepare(model, dataloader, optimizer, scheduler)

    model.train()
    logging.info("Starting training...")

    for epoch in range(args.num_epochs):
        for step, batch in tqdm(enumerate(dataloader, 1), dynamic_ncols=True, desc=f"Epoch {epoch + 1}/{args.num_epochs}", total=len(dataloader)):
            with accelerator.accumulate(model):
                loss, value_chosen, value_reject = model(**batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.v_head.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                global_step = epoch * len(dataloader) + step
                writer.add_scalar('RM/Loss', loss.item(), global_step)
                writer.add_scalar('RM/Chosen-Value', value_chosen, global_step)
                writer.add_scalar('RM/Rejected-Value', value_reject, global_step)
                writer.add_scalar('RM/Value-Gap', value_chosen - value_reject, global_step)

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
                print(color_text("\n" + center(f"Chosen Reward: {value_chosen:.2f}"), "cyan"))
                print(color_text("\n" + center(f"Rejected Reward: {value_reject:.2f}"), "green"))

    model.save_pretrained(args.output_dir)
    logging.info("Training completed and model saved.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
    sys.exit(0)
