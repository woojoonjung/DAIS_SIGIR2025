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
def _serialize_rough(json_obj, tokenizer):
        token_ids = tokenizer.encode(str(json_obj), add_special_tokens=False)
        tokenized = tokenizer.convert_ids_to_tokens(token_ids)
        tokenized = [token for token in tokenized if token != "'"]
        return " ".join(tokenized)

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

def _serialize(json_obj):
        """
        Serialize the JSON object with clear hierarchical key representation.
        """
        def serialize_recursive(obj, parent_key=""):
            parts = []
            if isinstance(obj, dict):
                parts.append("{")
                for k, v in obj.items():
                    full_key = f"{parent_key}.{k}" if parent_key else k
                    parts.append(f"{k}: {serialize_recursive(v, full_key)}")
                    parts.append(",")
                parts.append("}")
            elif isinstance(obj, list):
                parts.append("[")
                parts.append(", ".join([serialize_recursive(item, parent_key) for item in obj]))
                parts.append("]")
            else:
                parts.append(str(obj))
            return " ".join(parts)

        serialized = serialize_recursive(json_obj)
        return serialized

def _find_key_positions(serialized, json_obj, tokenizer, parent_key=""):
    tokenized = tokenizer(
        serialized,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"].squeeze(0).tolist()
    tokenized_serialized = tokenizer.convert_ids_to_tokens(input_ids)

    key_positions = {}
    current_position = 1  # After [CLS]

    def recurse_json(obj, parent_key=""):
        nonlocal current_position
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                tokenized_key = tokenizer.tokenize(key)

                key_start_pos = _find_token_indices(tokenized_serialized, tokenized_key, current_position)
                if key_start_pos is not None:
                    key_positions[full_key] = key_start_pos
                current_position = key_start_pos[-1] + 2 if key_start_pos else current_position + len(tokenized_key) + 2
                recurse_json(value, full_key)
                
        elif isinstance(obj, list):
            for i, element in enumerate(obj):
                recurse_json(element, f"{parent_key}[{i}]")
                
        else:
            current_position += len(tokenizer.tokenize(str(obj)))

    def _find_token_indices(sequence, tokens, start_index):
        for i in range(start_index, len(sequence) - len(tokens) + 1):
            if sequence[i:i+len(tokens)] == tokens:
                return list(range(i, i+len(tokens)))
        return None

    recurse_json(json_obj)
    return key_positions

############################## TOKENIZE ###################################

def tokenize_table(entry, model, tokenizer):
    if isinstance(tokenizer, TapasTokenizer):
        instance = {key: str(entry[key]) for key in entry.keys()}
        table = pd.DataFrame([instance])
        inputs = tokenizer(table=table, queries=["What is the missing value?"], padding="max_length", truncation=True, return_tensors="pt").to(device)
        tokenized_table = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return inputs, tokenized_table
    elif hasattr(model, "key_embedding"):
        serialized = _serialize(entry)
        tokenized_table = tokenizer.tokenize(serialized)[:512]
        inputs = tokenizer(
            serialized, 
            max_length=512, 
            truncation=True, 
            padding="max_length",
            return_tensors="pt",
            return_special_tokens_mask=True,
        ).to(device)
        return inputs, tokenized_table
    else:
        serialized = _serialize_rough(entry, tokenizer)
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


def mask_entry(entry, model, tokenizer, target="Key", mask_ratio=0.15):
    """
    Randomly mask a portion of a key or value in a tokenized JSON entry.
    - target="key" masks keys.
    - target="value" masks values.
    """
    inputs, tokenized_table = tokenize_table(entry, model, tokenizer)
    target_positions = [pos for pos in _find_positions(tokenized_table, tokenizer, entry, target) if pos < 512]
    
    if hasattr(model, "key_embedding"):
        serialized = _serialize(entry)
        key_positions = _find_key_positions(serialized, entry, tokenizer)
    else:
        key_positions = None

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

    return inputs, tokenized_table, masked_indices, labels, key_positions


def predict_masked_tokens(model, tokenizer, inputs, key_positions):
    """
    Returns predictions for masked tokens in TAPAS.
    """
    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(model, "key_embedding"):
        with torch.no_grad():
            outputs = model(**inputs, key_positions=[key_positions])
        predicted_ids = torch.argmax(outputs["logits"], dim=-1)

    else:
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_ids = torch.argmax(outputs.logits, dim=-1)

    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())

    return predicted_tokens


def evaluate_masked_prediction(data, target, model, tokenizer):
    correct = 0
    total = 0

    for i in range(len(data)):
        masked_inputs, _, masked_positions, labels, key_positions = mask_entry(data[i], model, tokenizer, target=target)
        predictions = predict_masked_tokens(model, tokenizer, masked_inputs, key_positions)
        
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

def get_table_embedding(entry, model, tokenizer, target=None):
    if target:
        entry.pop(target, None)
    inputs, _ = tokenize_table(entry, model, tokenizer)
    if hasattr(model, "key_embedding"):
        serialized = _serialize(entry)
        key_positions = _find_key_positions(serialized, entry, tokenizer)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, key_positions=[key_positions]) 
        last_hidden_state = outputs["hidden_states"][-1]
    else:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True) 
        last_hidden_state = outputs.hidden_states[-1]

    embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def get_table_cls_embedding(entry, model, tokenizer, key_positions=None, target=None):
    entry.pop(target, None)
    inputs, _ = tokenize_table(entry, model, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) 

    if hasattr(model, "key_embedding"):
        last_hidden_state = outputs["hidden_states"][-1][0, :]
    else:
        last_hidden_state = outputs.hidden_states[-1][:, 0, :]

    cls_embedding = last_hidden_state.squeeze().to("cpu").numpy()
    return cls_embedding
    

def prepare_Xy(path, model, tokenizer, target=None, seed=42):
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