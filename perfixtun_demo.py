from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, PrefixTuningConfig, TaskType

# model_name_or_path = "bigscience/mt0-large"
model_name_or_path = "bigscience/mt0-small"

# 对于这个模型，token_dim 必须要设置为 384，否则由于维度不匹配， model.generate 会报错
peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20, token_dim=384)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
# print(model)

model = get_peft_model(model, peft_config)
# print(model)
model.print_trainable_parameters() # trainable params: 122880 || all params: 300299648 || trainable%: 0.04091912888289499

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
outputs = model.generate(input_ids=inputs)
print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>



