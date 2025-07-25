from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./Qwen3-0.6B/")

sum(p.numel() for p in model.parameters())  # 596049920
print(model)

# Qwen3ForCausalLM(
#   (model): Qwen3Model(
#     (embed_tokens): Embedding(151936, 1024)
#     (layers): ModuleList(
#       (0-27): 28 x Qwen3DecoderLayer(
#         (self_attn): Qwen3Attention(
#           (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
#           (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
#           (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
#           (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
#           (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
#           (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
#         )
#         (mlp): Qwen3MLP(
#           (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
#           (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
#           (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
#         (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
#       )
#     )
#     (norm): Qwen3RMSNorm((1024,), eps=1e-06)
#     (rotary_emb): Qwen3RotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
# )
