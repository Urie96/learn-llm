from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B/")
tokens = tokenizer.tokenize("明月几时有，把酒问")
print(tokens)  # ['æĺİ', 'æľĪ', 'åĩł', 'æĹ¶', 'æľī', 'ï¼Į', 'æĬĬ', 'éħĴ', 'éĹ®']
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)  # [30858, 9754, 99195, 13343, 18830, 3837, 99360, 99525, 56007]

for input_id in input_ids:
    print(tokenizer.decode(input_id))
    # 明
    # 月
    # 几
    # 时
    # 有
    # ,
    # 把
    # 酒
    # 问
