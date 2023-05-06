from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType

model_name_or_path = "bert-base-cased"

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()   # trainable params: 1413636 || all params: 125468676 || trainable%: 1.1266844004953076

# TODO 训练中...

# 假装训练好了
text_column = "Tweet text"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
inputs = tokenizer(
    f'{text_column} : {"@nationalgridus I have no water and the bill is current and paid. Can you do something about this?"} Label : ',
    return_tensors="pt",
)

# 因为没有训练，所以实际上是瞎输出
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    # max_new_tokens=10,
    eos_token_id=102
)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
