import torch


def pad_to_left(input_ids, max_length, pad_token_id=0):
    """
    Pads the input_ids to the left with zeros up to max_length.
    
    Args:
        input_ids (list): List of token IDs to be padded.
        max_length (int): The desired length after padding.
    
    Returns:
        list: Padded list of token IDs.
    """
    if len(input_ids) >= max_length:
        input_ids = input_ids[:max_length]
    else:
        input_ids = [pad_token_id] * (max_length - len(input_ids)) + input_ids
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = (input_ids != pad_token_id).long()
    return input_ids, attention_mask


@torch.no_grad()
def generate(model, input_ids, attention_mask, pad=0, eos=2):
    prompt_length = input_ids.shape[1]
    generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            pad_token_id=pad,
            eos_token_id=eos,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.5,
        )
    length = (generated_ids[:, prompt_length:] != pad).sum(dim=1)
    return generated_ids[length > 0]


def calculate_action_logsoftmax(logits, chosen_ids):
    log_probs = logits.log_softmax(dim=-1)
    return log_probs.gather(2, chosen_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def generate_batch_data(model, model_ref, model_critic, model_reward, input_ids, attention_mask, pad=0, eos=2):
    model.eval()
    model_device = model.device
    ref_device = model_ref.device
    reward_device = model_reward.device
    
    generated_ids = generate(model, input_ids, attention_mask, pad, eos)
    if len(generated_ids) == 0:
        return None, None, None, None, None, None
    generated_attention_mask = (generated_ids != pad).long()
    
    logits_old = model(generated_ids, generated_attention_mask).logits
    log_prob_old = calculate_action_logsoftmax(logits_old[:, :-1], generated_ids[:, 1:])
    value_old = model_critic(generated_ids, attention_mask=generated_attention_mask)
    reward = model_reward.get_reward(generated_ids.to(reward_device), values=value_old.to(reward_device)).to(model_device)
    
    logits_ref = model_ref(generated_ids.to(ref_device), generated_attention_mask.to(ref_device)).logits.to(model_device)
    log_prob_ref = calculate_action_logsoftmax(logits_ref[:, :-1], generated_ids[:, 1:])
    model.train()
    return generated_ids, generated_attention_mask, log_prob_old, value_old, reward, log_prob_ref


def get_eos_position(generated_ids, eos=2):
    # generated_ids: [batch_size, seq_len]
    batch_size, seq_len = generated_ids.shape
    eos_positions = []
    for i in range(batch_size):
        pos = (generated_ids[i] == eos).nonzero(as_tuple=True)
        eos_positions.append(pos[0].item() if pos[0].numel() > 0 else seq_len - 1)
    return eos_positions


@torch.no_grad()
def calculate_reward_with_kl(end, log_prob_old, log_prob_ref, reward, coeff=0.1):
    """
    Calculate the reward with KL divergence penalty.
    KL-Reward for each non-eos token with index `idx` is calculated as
        -0.1 * (log_prob_old[idx] - log_prob_ref[idx])
    For eos token with index `end_pos`, the KL-Reward is calculated as
        -0.1 * (log_prob_old[end_pos] - log_prob_ref[end_pos]) + reward.clamp(-5, 5)
    """
    kl = -coeff * (log_prob_old - log_prob_ref)
    reward_kl = kl.clone()
    
    for idx, end_pos in enumerate(end):
        if end_pos >= reward_kl.shape[1]:
            end_pos = -1
        reward_kl[idx, end_pos] += reward[idx].clamp(-5, 5)
    
    return kl, reward_kl


@torch.no_grad()
def calculate_td_delta(reward_kl, value_old, gamma=1.0, prompt_length=0):
    V_s = value_old[:, :-1]
    V_next = value_old[:, 1:]

    td_delta = reward_kl + gamma * V_next - V_s

    return td_delta[:, prompt_length - 1:]


@torch.no_grad()
def calculate_advantage(td_delta, lmbda=0.95, gamma=1.0):
    advantage = []
    adv = 0.0
    for delta in td_delta.flip(dims=[1]).unbind(dim=1):
        adv = lmbda * gamma * adv + delta
        advantage.append(adv)
    advantage.reverse()
    return torch.stack(advantage, dim=1)
