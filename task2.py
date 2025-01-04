import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import openai
import torch

df = pd.read_csv("./trainingset-ideology-power/power/power-es-train.tsv", sep="\t")

train_data, test_data = train_test_split(
    df, test_size=0.1, stratify=df['label'], random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

def preprocess_data(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True
    )

tokenized_train = Dataset.from_pandas(train_data).map(preprocess_data, batched=True)
tokenized_test_data = Dataset.from_pandas(test_data).map(preprocess_data, batched=True)

def remove_unused_columns(dataset):
    columns_to_remove = [col for col in ["text", "__index_level_0__"] if col in dataset.column_names]
    return dataset.remove_columns(columns_to_remove)

tokenized_train = remove_unused_columns(tokenized_train)
tokenized_train.set_format("torch")

tokenized_test_data = remove_unused_columns(tokenized_test_data)
tokenized_test_data.set_format("torch")

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
    max_grad_norm=1.0,
    warmup_steps=500,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
)
trainer.train()

print(classification_report(test_data['label'].values, np.argmax(trainer.predict(tokenized_test_data).predictions, axis=-1)
))

test_data_v2 = test_data.drop(columns=['label']).to_csv('./test_v2.tsv', sep='\t', index=False) 
print("Classification Report:\n", classification_report(test_data['label'], pd.read_csv("./results_v2.csv")['label']))

openai.api_key = "" # I deleted my key for security purposes
if not openai.api_key:
    raise ValueError("OpenAI API key is not set!")

def classify_text_gpt3(speech):
    prompt = f"""You are an expert political analyst. 
        Based on the following parliamentary speech excerpt, determine whether the speaker's party is currently governing (0) 
        or in opposition (1). 
        Provide only the number 0 or 1 as your answer and nothing else.\n\nSpeech: "{speech}"\nYour classification:"""

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1,
        temperature=0.0
    )
    return response["choices"][0]["text"].strip()

for index, row in tokenized_test_data.iterrows():
    print(classify_text_gpt3(row['text']))

