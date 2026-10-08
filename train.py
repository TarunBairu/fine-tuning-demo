from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
import pandas as pd


# -----------------------------
# 1. Load Dataset
# -----------------------------

df = pd.read_csv("data/train.csv")

print("Dataset:")
print(df)


# -----------------------------
# 2. Create Label Mapping
# -----------------------------

labels = sorted(df["label"].unique())

label2id = {
    label: index
    for index, label in enumerate(labels)
}

id2label = {
    index: label
    for label, index in label2id.items()
}

print("\nLabels:")
print(label2id)


df["label"] = df["label"].map(label2id)


# -----------------------------
# 3. Split Dataset
# -----------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# -----------------------------
# 4. Convert to Hugging Face Dataset
# -----------------------------

from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# -----------------------------
# 5. Load Pre-trained Model
# -----------------------------

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label
)


# -----------------------------
# 6. Tokenize Dataset
# -----------------------------

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# -----------------------------
# 7. Training Configuration
# -----------------------------

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True
)


# -----------------------------
# 8. Create Trainer
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)


# -----------------------------
# 9. Fine-Tune
# -----------------------------

print("\nStarting fine-tuning...")

trainer.train()


# -----------------------------
# 10. Save Model
# -----------------------------

trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("\nFine-tuning completed!")
print("Model saved in: ./fine_tuned_model")