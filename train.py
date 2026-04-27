from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer

labels = [
    "loneliness","social_isolation","sadness","grief","anxiety","fear",
    "confusion","memory_loss","disorientation","frustration","helplessness",
    "boredom","fatigue","sleep_issue","health_concern","medication_issue",
    "appetite_loss","happiness","gratitude","neutral"
]

# =========================
# LOAD DATASET
# =========================
dataset = load_dataset("json", data_files="data.json")

# 🔥 IMPORTANT : renommer label → labels
dataset = dataset.rename_column("label", "labels")

# =========================
# TOKENIZER
# =========================
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize)

# 🔥 IMPORTANT : format torch
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# =========================
# MODEL
# =========================
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels)   # = 20
)

# =========================
# TRAINING CONFIG
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    save_strategy="no"
)

# =========================
# TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# SAVE
# =========================
model.save_pretrained("careai_model")
tokenizer.save_pretrained("careai_model")

print("✅ Model trained and saved in careai_model/")