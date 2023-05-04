from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# model_name_or_path = "bigscience/mt0-large"
model_name_or_path = "bigscience/mt0-small"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
print(model)

model = get_peft_model(model, peft_config)
print(model)
model.print_trainable_parameters() # trainable params: 344064 || all params: 300520832 || trainable%: 0.11448923447676333

# once forward--权重没有估计，也没有merge
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
outputs = model.generate(input_ids=inputs)
print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>


# inference mode--这样只是权重固定，但是没有 merge
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=True, r=8, lora_alpha=32, lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # trainable params: 0 || all params: 300520832 || trainable%: 0.0

# 一旦设置 merge_adapter 就会自动合并了
model.base_model.merge_adapter()

# once forward
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
outputs = model.generate(input_ids=inputs)
print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>


