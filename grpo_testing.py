import torch
import os
from transformers import AutoTokenizer
from GRPO.dataset import GSM8kDataset
# from GRPO.pot_reward import run_code_from_text
from GRPO.cot_reward import extract_answer, FORMAT_PATTERN
from utils import load_model, GRPO_MODEL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_response(model, input_ids, attention_mask, tokenizer):
    generated_ids = model.generate(
        input_ids=input_ids[None],
        attention_mask=attention_mask[None],
        max_new_tokens=512,
        do_sample=True,
        temperature=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=False
    )[0][256:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def evaluate(response, answer):
    _, result = extract_answer(response)
    is_accurate = False
    is_formatted = FORMAT_PATTERN.fullmatch(response) is not None
    if result is not None:
        try:
            result = float(result)
            answer = float(answer)
            if abs(result - answer) < 1e-5:
                is_accurate = True
        except Exception:
            ...
    return is_accurate, is_formatted


tokenizer = AutoTokenizer.from_pretrained(GRPO_MODEL_NAME, trust_remote_code=True, padding_side='left',
                                          local_files_only=True)
dataset = GSM8kDataset(tokenizer=tokenizer, split='test', max_length=256)
model = load_model(GRPO_MODEL_NAME).eval().cuda()
model_grpo = load_model('model/grpo').eval().cuda()

acc = 0
format_acc = 0
acc_grpo = 0
format_acc_grpo = 0
total = 0
with torch.inference_mode():
    for data in dataset:
        total += 1
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        answer = data['answer']
        response0 = generate_response(model, input_ids, attention_mask, tokenizer)
        is_acc, is_formatted = evaluate(response0, answer)
        if is_acc:
            acc += 1
        if is_formatted:
            format_acc += 1
        response1 = generate_response(model_grpo, input_ids, attention_mask, tokenizer)
        is_acc, is_formatted = evaluate(response1, answer)
        if is_acc:
            acc_grpo += 1
        if is_formatted:
            format_acc_grpo += 1
        print(f'Evaluate: {total} / {len(dataset)} | '
              f'Qwen Accuracy: {acc / total:.4f} Formatted: {format_acc / total:.4f} | '
              f'GRPO Accuracy: {acc_grpo / total:.4f} Formatted: {format_acc_grpo / total:.4f}', end='\r')
print(f'\nQwen Accuracy: {acc / total:.4f} Formatted: {format_acc / total:.4f} '
      f'GRPO Accuracy: {acc_grpo / total:.4f} Formatted: {format_acc_grpo / total:.4f}')
