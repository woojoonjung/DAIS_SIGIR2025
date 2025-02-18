import os
import json
import random
import pandas as pd
import numpy as np
from io import StringIO

from transformers import (
    BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling,
    TapasTokenizer, TapasForMaskedLM,
    AdamW, get_scheduler
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU is available and will be used.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

############################## SERIALIZE ###################################

def _serialize_vanilla(json_obj, parent_key="", sep="."):
    """
    Serialize a JSON object into a string format suitable for tokenization, handling nested structures.
    """
    serialized = []
    for key, value in json_obj.items():
        full_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            serialized.append(_serialize_vanilla(value, parent_key=full_key, sep=sep))
        elif isinstance(value, list):
            list_content = ", ".join([str(v) if not isinstance(v, dict) else _serialize_vanilla(v, parent_key=full_key, sep=sep) for v in value])
            serialized.append(f"{full_key} is [{list_content}]")
            serialized.append(",")
        else:
            serialized.append(f"{full_key} is {value}")
            serialized.append(",")
    return " ".join(serialized)

def _serialize(tokenizer, json_obj):
    tokenized = tokenizer.tokenize(str(json_obj))
    tokenized = [token for token in tokenized if token != "'"]
    return " ".join(tokenized)

############################## TOKENIZE ###################################

def tokenize_table(entry, tokenizer):
    if isinstance(tokenizer, TapasTokenizer):
        instance = {key: str(entry[key]) for key in entry.keys()}
        table = pd.DataFrame([instance])
        inputs = tokenizer(table=table, queries=["What is the missing value?"], padding="max_length", truncation=True, return_tensors="pt").to(device)
        tokenized_table = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return inputs, tokenized_table
    else:
        serialized = _serialize(tokenizer, entry)
        tokenized_table = tokenizer.tokenize(serialized)[:512]
        inputs = tokenizer(serialized, padding="max_length", truncation=True, return_tensors="pt").to(device)
        return inputs, tokenized_table

###########################################################################
###########################################################################

###################### MASK PREDICTION EXPERIMENT #########################

def _find_positions(tokenized_table, tokenizer, json_obj, target="Key"):
    """
    Find positions of tokens that correspond to keys or values in a JSON object.
    - target="key" finds key token positions.
    - target="value" finds value token positions.
    """
    positions = []
    for key, value in json_obj.items():
        target_text = str(key) if target == "Key" else str(value)
        target_tokens = tokenizer.tokenize(text=target_text)

        # Find matching token positions in the full tokenized table
        for i in range(len(tokenized_table) - len(target_tokens) + 1):
            if tokenized_table[i : i + len(target_tokens)] == target_tokens:
                positions.extend(range(i, i + len(target_tokens)))

    return positions


def mask_entry(entry, tokenizer, target="Key", mask_ratio=0.15):
    """
    Randomly mask a portion of a key or value in a tokenized JSON entry.
    - target="key" masks keys.
    - target="value" masks values.
    """
    inputs, tokenized_table = tokenize_table(entry, tokenizer)
    target_positions = [pos for pos in _find_positions(tokenized_table, tokenizer, entry, target) if pos < 512]

    # Select a subset of tokens to mask
    num_masked = max(3, int(mask_ratio * len(target_positions)))
    masked_indices = random.sample(target_positions, min(num_masked, len(target_positions)))

    # Create labels tensor for (same size as input_ids, filled with -100)
    labels = inputs["input_ids"].clone()
    labels.fill_(-100)

    for idx in masked_indices:
        labels[0, idx] = inputs["input_ids"][0, idx] 
        inputs["input_ids"][0, idx] = 103 
        tokenized_table[idx] = "[MASK]"

    return inputs, tokenized_table, masked_indices, labels


def predict_masked_tokens(model, tokenizer, inputs):
    """
    Returns predictions for masked tokens in TAPAS.
    """
    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(model, "key_embedding"):
        predicted_ids = torch.argmax(outputs["logits"], dim=-1)
    else:
        predicted_ids = torch.argmax(outputs.logits, dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())

    return predicted_tokens


def evaluate_masked_prediction(data, target, model, tokenizer):
    correct = 0
    total = 0

    for i in range(len(data)):
        masked_inputs, _, masked_positions, labels = mask_entry(data[i], tokenizer, target=target)
        predictions = predict_masked_tokens(model, tokenizer, masked_inputs)
        
        for idx in masked_positions:
            true_tokens = tokenizer.convert_ids_to_tokens(labels[0])
            if true_tokens[idx] == predictions[idx]:
                correct += 1
            total += 1

    accuracy = correct / total
    
    print(f"Correct / Total: {correct}/{total}")
    print(f"Model Accuracy on Masked {target} Prediction: {accuracy:.4f}%")

###########################################################################
###########################################################################

###################### CLASSIFICATION EXPERIMENT ##########################

def get_table_embedding(entry, model, tokenizer, target='class'):
    entry.pop(target, None)
    inputs, _ = tokenize_table(entry, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) 

    if hasattr(model, "key_embedding"):
        last_hidden_state = outputs["hidden_states"][-1]
    else:
        last_hidden_state = outputs.hidden_states[-1]

    embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def get_table_cls_embedding(entry, model, tokenizer, target='class'):
    entry.pop(target, None)
    inputs, _ = tokenize_table(entry, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) 

    if hasattr(model, "key_embedding"):
        last_hidden_state = outputs["hidden_states"][-1][0, :]
    else:
        last_hidden_state = outputs.hidden_states[-1][:, 0, :]

    cls_embedding = last_hidden_state.squeeze().to("cpu").numpy()
    return cls_embedding
    

def prepare_Xy(path, model, tokenizer, target='class', seed=42):
    # Prepare data
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    df = pd.read_csv(StringIO(''.join(lines)))
    data = df.to_dict(orient="records")
    y = df[target].values
    
    # Split first (before embedding extraction)
    train_data, test_data, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=seed)

    # Extract embeddings
    X_train = np.array([get_table_embedding(entry, model, tokenizer, target) for entry in train_data])
    X_test = np.array([get_table_embedding(entry, model, tokenizer, target) for entry in test_data])

    return X_train, X_test, y_train, y_test

def train_eval_rf(X_train, X_test, y_train, y_test, seed=42):
    # Train the classifier
    clf = RandomForestClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)

    if len(set(y_test)) == 2:
        pos_label = random.choice(y_test)
        metrics = {
            "precision": precision_score(y_test, y_pred, pos_label=pos_label, average="binary", zero_division=0),
            "recall": recall_score(y_test, y_pred, pos_label=pos_label, average="binary", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, pos_label=pos_label, average="binary", zero_division=0),
        }
    else:
        metrics = {
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }
    
    return metrics

###########################################################################
###########################################################################