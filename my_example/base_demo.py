from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "bigscience/mt0-small"  # mt5 的 funtune 版本
tokenizer_name_or_path = "bigscience/mt0-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
# model = AutoModelForCausalLM.from_pretrained("internlm/internlm-7b", trust_remote_code=True)
print(model)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 因为输入是 prompt + text，所以实际上是条件生成模型
inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")

outputs = model.generate(input_ids=inputs)
print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>
