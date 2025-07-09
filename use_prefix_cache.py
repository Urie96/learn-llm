from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B/")
model = AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B/")


def create_prefix_cache(prefix):
    text = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.past_key_values, input_ids.shape[1]


def generate(input_ids, attention_mask, past_key_values):
    gen_text = ""
    for _ in range(500):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values)
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        decoded_token = tokenizer.decode(next_token_id)
        if decoded_token == "<|im_end|>":
            break
        past_key_values = outputs.past_key_values
        input_ids = torch.tensor([[next_token_id]])
        attention_mask = torch.cat([attention_mask, torch.tensor([[1]])], dim=1)
        gen_text += decoded_token
    return gen_text


def generate_using_prefix_cache(msg, prefix_key_values, prefix_token_length):
    suffix_text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    user_inputs = tokenizer(suffix_text, return_tensors="pt")
    input_ids = user_inputs["input_ids"]
    attention_mask = torch.ones(
        1, prefix_token_length + input_ids.shape[1], dtype=torch.int64
    )

    return generate(input_ids, attention_mask, prefix_key_values)


prefix_key_values, prefix_token_length = create_prefix_cache([{"role": "system", "content": "你是一个英语翻译专家，请帮我把中文翻译为英语"}])
msg = [{"role": "user", "content": "小荷健康"}]
print(generate_using_prefix_cache(msg, prefix_key_values, prefix_token_length))

# 输出：
# small leaf health
