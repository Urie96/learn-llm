from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B/")
inputs = tokenizer("明月几时有，把酒问", return_tensors="pt")

input_ids = inputs["input_ids"]

model = AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B/")

for i in range(10):
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    # 获取最后一个token的logits，用于预测下一个token
    next_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    print(tokenizer.decode(next_token_id), end="")

    # 将新生成的token_id添加到input_ids用来继续生成下一个token
    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)

# 输出：
# 青天。对酒当歌，人生几
