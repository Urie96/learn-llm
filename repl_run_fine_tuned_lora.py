from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_qwen3 = AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B/")
peft_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B/"),
    "./Qwen3-0.6B-fine-tuned-lora",
)
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
        print("微调模型: ", complete(peft_model, user_input))
        print("原Qwen3模型:", complete(model_qwen3, user_input))
    except Exception:
        break
