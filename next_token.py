from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B/")
inputs = tokenizer("明月几时有，把酒问", return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B/")
sum(p.numel() for p in model.parameters())  # 596049920
sum(p.numel() for p in model.parameters() if p.requires_grad)
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
)
next_token_logits = outputs.logits[0, -1, :]
next_token_id = torch.argmax(next_token_logits).item()
print(tokenizer.decode(next_token_id))  # "新"
