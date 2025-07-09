from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer


tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B/")

datasets = load_dataset(
    "text",
    data_files={
        "train": "./datasets/train.txt",
        "validation": "./datasets/validation.txt",
    },
)
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

training_args = TrainingArguments(
    "Qwen3-0.6B-fine-tuned",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

trainer.train()

trainer.save_model("Qwen3-0.6B-fine-tuned")
