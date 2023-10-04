import pandas as pd
import os
import numpy as np
import transformers
import datasets
import evaluate

path_to_data = "data_all.csv"
data = pd.read_csv(path_to_data)
unique_labels = data['label'].unique()
emoji2int = {
    emoji: i for i, emoji in enumerate(unique_labels)
}
labels = data['label'].apply(lambda x: emoji2int[x])
data_emojis = data.drop(["total", "text", "chat_type"], axis=1)
texts = data["text"]
model_name = "bert-base-multilingual-cased"

dataset = datasets.Dataset.from_dict({"text": texts, "label": labels})

model = transformers.BertForSequenceClassification.from_pretrained(
    model_name, num_labels=labels.nunique()
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
