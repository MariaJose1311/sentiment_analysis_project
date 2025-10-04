from transformers import BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from preprocess import load_and_split_dataset, tokenize_dataset, map_labels_to_int
from utils import set_seed, compute_metrics
from config import DATA_PATH, MODEL_DIR, DEVICE

set_seed(42)

# -------------------
# Dataset
# -------------------
train_ds, val_ds = load_and_split_dataset(DATA_PATH)
train_ds, tokenizer = tokenize_dataset(train_ds, f"{MODEL_DIR}/fase1")
val_ds, _ = tokenize_dataset(val_ds, f"{MODEL_DIR}/fase1")
train_ds = map_labels_to_int(train_ds)
val_ds = map_labels_to_int(val_ds)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------
# Modelo
# -------------------
model = BertForSequenceClassification.from_pretrained(f"{MODEL_DIR}/fase1")
model.to(DEVICE)

# Congelar primeras 8 capas
for param in model.bert.encoder.layer[:8].parameters():
    param.requires_grad = False

# -------------------
# Entrenamiento Fase 2
# -------------------
args2 = TrainingArguments(
    output_dir=f"{MODEL_DIR}/fase2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=20,
    report_to='none'
)

trainer2 = Trainer(
    model=model,
    args=args2,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer2.train()
trainer2.save_model(f"{MODEL_DIR}/fase2")
tokenizer.save_pretrained(f"{MODEL_DIR}/fase2")
print("Fase 2 completada ✅")
