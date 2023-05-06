import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import (
    get_peft_model,
    PromptEncoderConfig,
)

# encoder_hidden_size: the hidden size of the encoder used to optimize the prompt parameters
peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)

# RoBERTa是在论文《RoBERTa: A Robustly Optimized BERT Pretraining Approach》中被提出的。此方法属于BERT的强化版本，也是BERT模型更为精细的调优版本
model_name_or_path = "roberta-base"

# 基于 base 模型构建二分类任务层
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # trainable params: 1413636 || all params: 125468676 || trainable%: 1.1266844004953076

# TODO 训练中...

# 假装微调好了这个二分类模型
classes = ["not equivalent", "equivalent"]

sentence1 = "Coast redwood trees are the tallest trees on the planet and can grow over 300 feet tall."
sentence2 = "The coast redwood trees, which can attain a height of over 300 feet, are the tallest trees on earth."

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
inputs = tokenizer(sentence1, sentence2, truncation=True, padding="longest", return_tensors="pt")
# 两句话拼接，中间用特定分隔符区分句子 (1, 47)
with torch.no_grad():
    outputs = model(**inputs).logits

# not equivalent: 46%
# equivalent: 54%
paraphrased_text = torch.softmax(outputs, dim=1).tolist()[0]
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrased_text[i] * 100))}%")

