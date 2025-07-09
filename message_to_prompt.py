from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B/")
text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "请帮我翻译为英语"},
        {"role": "user", "content": "小荷健康"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

print(text)
