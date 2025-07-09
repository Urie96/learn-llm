from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B/")
inputs = tokenizer("明月几时有，把酒问", return_tensors="pt")

input_ids = inputs["input_ids"]
attention_mask = torch.ones_like(input_ids)
past_key_values = None  # 保存kv cache

model = AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B/")

for _ in range(10):
    # 如果有kv cache，只需要输入最后一个token即可
    outputs = model(
        input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
        attention_mask=torch.ones_like(input_ids),
        past_key_values=past_key_values,
    )
    past_key_values = outputs.past_key_values  # 更新为这次产生的kv cache
    next_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    print(tokenizer.decode(next_token_id), end="")

    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)

# 输出（和不使用 KV cache 时是一样的）：
# 新，持续创新，是企业发展的核心动力
