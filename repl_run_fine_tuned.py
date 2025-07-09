from transformers import AutoModelForCausalLM, AutoTokenizer

model_fine_tuned = AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B-fine-tuned")
model_base = AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B/")

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B/")


def complete(model, user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


while True:
    try:
        user_input = input(">>> ")
        print("微调模型: ", complete(model_fine_tuned, user_input))
        print("原Qwen3模型:", complete(model_base, user_input))
    except Exception:
        break
