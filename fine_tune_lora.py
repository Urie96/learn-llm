from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from peft import LoraConfig, get_peft_model, TaskType


datasets = load_dataset(
    "text",
    data_files={
        "train": "./datasets/lora_train.txt",
        "validation": "./datasets/validation.txt",
    },
)
tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B/")


tokenized_datasets = datasets.map(
    lambda x: tokenizer(x["text"]), remove_columns=["text"]
)


block_size = 128


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
)

model = AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B/", device_map="cpu")

lora_config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,  # dropout率
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数数量

training_args = TrainingArguments(
    "yangrui-this-word-lora",
    eval_strategy="steps",
    eval_steps=200,
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    num_train_epochs=10,
    warmup_steps=100,
    weight_decay=0.001,
    logging_steps=50,
    save_steps=500,
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],  # type: ignore
    eval_dataset=lm_datasets["validation"],  # type: ignore
)

trainer.train()

model.save_pretrained("my_lora_fine_tuned_model")
