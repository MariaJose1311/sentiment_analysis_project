from transformers import BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from preprocess import load_and_split_dataset, tokenize_dataset, map_labels_to_int
from utils import set_seed, compute_metrics
from config import DATA_PATH, MODEL_DIR, DEVICE

set_seed(42)

# -------------------
# Dataset
# -------------------
train_ds, val_ds = load_and_split_dataset(DATA_PATH)
train_ds, tokenizer = tokenize_dataset(train_ds, "dccuchile/bert-base-spanish-wwm-cased")
val_ds, _ = tokenize_dataset(val_ds, "dccuchile/bert-base-spanish-wwm-cased")
train_ds = map_labels_to_int(train_ds)
val_ds = map_labels_to_int(val_ds)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------
# Modelo
# -------------------
num_labels = 2
model = BertForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", num_labels=num_labels)
model.to(DEVICE)

# -------------------
# Entrenamiento Fase 1
# -------------------
args = TrainingArguments(
    output_dir=f"{MODEL_DIR}/fase1",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(f"{MODEL_DIR}/fase1")
tokenizer.save_pretrained(f"{MODEL_DIR}/fase1")
print("Fase 1 completada ✅")
