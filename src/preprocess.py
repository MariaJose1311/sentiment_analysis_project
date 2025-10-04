import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer

def load_and_split_dataset(path, test_size=0.3, seed=42):
    df = pd.read_excel(path)
    df['label'] = df['rating'].apply(lambda r: "no positiva" if r <= 3 else "positiva")
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=seed)
    return Dataset.from_pandas(train_df[['text','label']].reset_index(drop=True)), \
           Dataset.from_pandas(val_df[['text','label']].reset_index(drop=True))

def tokenize_dataset(ds, tokenizer_path, max_length=128):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding=False, max_length=max_length)

    ds = ds.map(tokenize, batched=True, batch_size=16)
    return ds, tokenizer

def map_labels_to_int(ds):
    # Crea la columna 'labels' con valores 0/1
    ds = ds.map(lambda x: {"labels": 1 if x["label"] == "positiva" else 0})
    # Remueve la columna original 'label'
    if "label" in ds.column_names:
        ds = ds.remove_columns(["label"])
    return ds
