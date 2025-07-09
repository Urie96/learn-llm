from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

model = AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B/")

input_ids = tokenizer(["猫", "狗", "热"])["input_ids"]
embeddings = model.get_input_embeddings()  # [151936, 1024]

a = embeddings(torch.tensor([input_ids]))[0]
print(F.cosine_similarity(a[0], a[1]))  # 猫 狗 0.4233
print(F.cosine_similarity(a[0], a[2]))  # 猫 热 0.0591
print(F.cosine_similarity(a[1], a[2]))  # 狗 热 0.0231
