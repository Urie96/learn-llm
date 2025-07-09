from transformers.pipelines import pipeline
import torch

print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

pipe = pipeline(task="text-generation", model="./Qwen3-0.6B/", do_sample=False)
print(pipe("明月几时有，把酒问"))
