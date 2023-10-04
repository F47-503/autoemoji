import pandas as pd
import os
import numpy as np
import transformers
import datasets
import evaluate

data = pd.read_csv("data_all.csv")
data_emojis = data.drop(["total", "text"], axis=1)
emojis = data_emojis.columns
data_np = data_emojis.to_numpy()
labels = np.argmax(data_np, axis=-1)
texts = data["text"]
model_name = "DeepPavlov/rubert-base-cased"

dataset = datasets.Dataset.from_dict({"text": texts, "label": labels})

model = transformers.BertForSequenceClassification.from_pretrained(
    model_name, num_labels=data_np.shape[1]
)
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)


def tokenizer_func(x):
    return tokenizer(x["text"], max_length=128, truncation=True, padding="max_length")


tokenized_dataset = dataset.map(tokenizer_func).shuffle(seed=503)
splitted_dataset = tokenized_dataset.train_test_split(test_size=0.2)

metric = evaluate.load("accuracy")


def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = transformers.TrainingArguments(
    output_dir=model_name,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=splitted_dataset["train"],
    eval_dataset=splitted_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
