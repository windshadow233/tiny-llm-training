import re


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FORMAT_PATTERN = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)
NUMBER_PATTERN = re.compile(r'-?\d+(?:\.\d+)?')


def extract_answer(text):
    answer_text = ANSWER_PATTERN.search(text)
    if answer_text:
        ans = answer_text.group(1).strip().replace(',', '')
        if NUMBER_PATTERN.fullmatch(ans):
            return True, ans
        numbers = NUMBER_PATTERN.findall(ans)
        if numbers:
            return False, numbers[-1]
    return False, None


def format_score(text):
    reward = 0
    think_start_count = text.count("<think>")
    think_end_count = text.count("</think>")
    answer_start_count = text.count("<answer>")
    answer_end_count = text.count("</answer>")
    
    if think_start_count == 1:
        reward += 0.2
    elif think_start_count > 1:
        reward -= 0.5 * (think_start_count - 1)
        
    if think_end_count == 1:
        reward += 0.2
    elif think_end_count > 1:
        reward -= 0.5 * (think_end_count - 1)
    
    if answer_start_count == 1:
        reward += 0.2
    elif answer_start_count > 1:
        reward -= 0.5 * (answer_start_count - 1)
        
    if answer_end_count == 1:
        reward += 0.2
    elif answer_end_count > 1:
        reward -= 0.5 * (answer_end_count - 1)
    
    return reward


def accuracy_reward(prompt, responses, answer):
    answers = [extract_answer(r) for r in responses]
    scores = []
    for is_pure_number, result in answers:
        score = 0.0
        if is_pure_number:
            score += 0.5
        if result is not None:
            try:
                result = float(result)
                answer = float(answer)
                if abs(result - answer) < 1e-5:
                    score += 2.0
            except ValueError:
                ...
        scores.append(score)
    return scores


def strict_format_reward(prompt, responses, answer):
    matches = [FORMAT_PATTERN.fullmatch(response) for response in responses]
    return [1.0 if match else 0.0 for match in matches]


def soft_format_reward(prompt, responses, answer):
    return [format_score(response) for response in responses]



REWARD_FCNS = {
    'Accuracy': accuracy_reward,
    'Strict Format': strict_format_reward,
    'Soft Format': soft_format_reward,
}
