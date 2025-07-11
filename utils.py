import os
import shutil
from transformers import AutoModelForCausalLM
from peft import PeftConfig, PeftModel


MODEL_NAME = 'm-a-p/CT-LLM-Base'
# GRPO_MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
GRPO_MODEL_NAME = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'


def color_text(text, color):
    COLOR_CODES = {
        'black': '\033[90m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    return f"{COLOR_CODES.get(color, '')}{text}{COLOR_CODES['reset']}"


def center(text, fillchar="="):
    width = shutil.get_terminal_size().columns
    wrapped = text.center(width, fillchar)
    return wrapped


def count_params(model):
    count_all = [i.numel() for i in model.parameters()]
    count_all = sum(count_all) / 1_000_000_000

    count_require = [i.numel() for i in model.parameters() if i.requires_grad]
    count_require = sum(count_require) / 1_000_000_000

    ratio = count_require / count_all

    return {
        'count_require': count_require,
        'count_all': count_all,
        'ratio': ratio
    }


def load_model(path, torch_dtype=None):
    if os.path.exists(os.path.join(path, 'adapter_config.json')):
        # LoRA adapter-only
        peft_config = PeftConfig.from_pretrained(path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
    return model