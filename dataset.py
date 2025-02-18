import os
import pandas as pd
import json
import torch
import random
import hashlib
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
from dataclasses import dataclass
import fasttext
from io import StringIO

SEED = 42
random.seed(SEED)

model_path = './lid.176.bin'
model = fasttext.load_model(model_path)

@dataclass
class JSONDataCollator(DataCollatorForLanguageModeling):
    mask_keys_only: bool = True  # Start by masking only keys
    key_mask_probability: float = 0.2  # Probability for masking keys
    nonkey_mask_probability: float = 0.15  # Probability for masking non-keys
    mask_replace_prob: float = 0.8  # Probability of replacing masked tokens with [MASK]
    random_replace_prob: float = 0.1  # Probability of replacing masked tokens with random tokens
    hybrid_epochs: int = 6  # Number of epochs for hybrid masking

    def __init__(self, tokenizer, mlm_probability=0.15, **kwargs):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.current_epoch = 0  # Tracks the current epoch
        self.hybrid_mode = True  # Start with the hybrid mode
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_epoch(self, current_epoch: int):
        """
        Set the current epoch and determine masking mode.
        """
        self.current_epoch = current_epoch
        # Determine whether to use hybrid or simultaneous masking
        if self.current_epoch < self.hybrid_epochs:
            self.hybrid_mode = True
            self.mask_keys_only = self.current_epoch % 2 == 0  # Alternate masking modes within hybrid_epochs
        else:
            self.hybrid_mode = False
            self.mask_keys_only = False  # Use simultaneous masking after hybrid_epochs

    def torch_mask_tokens(self, inputs, special_tokens_mask=None, key_positions=None):
        """
        Mask tokens with different probabilities for keys and non-keys, supporting hybrid and simultaneous masking modes.
        """
        labels = inputs.clone()

        # Base probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Convert special_tokens_mask to boolean, if provided
        if special_tokens_mask is not None:
            special_tokens_mask = special_tokens_mask.bool()

        # Apply hybrid or simultaneous masking
        if self.hybrid_mode and key_positions:
            for batch_idx, key_dict in enumerate(key_positions):
                all_positions = set(range(probability_matrix.size(1)))  # All possible positions
                key_positions_set = set()  # Track all positions specified as keys

                for key, positions in key_dict.items():
                    key_positions_set.update(positions)

                # Apply masking based on the current mode
                if self.mask_keys_only:
                    # print("Current mode: Masking keys")
                    for pos in key_positions_set:  # Mask only keys
                        probability_matrix[batch_idx, pos] = self.key_mask_probability
                else:
                    # print("Current mode: Masking non-keys")
                    non_key_positions = all_positions - key_positions_set
                    for pos in non_key_positions:  # Mask only non-keys
                        probability_matrix[batch_idx, pos] = self.nonkey_mask_probability
        else:  # Simultaneous masking for keys and non-keys
            # print("Current mode: Simultaneous masking")
            if key_positions:
                for batch_idx, key_dict in enumerate(key_positions):
                    all_positions = set(range(probability_matrix.size(1))) 
                    key_positions_set = set()

                    for key, positions in key_dict.items():
                        for pos in positions:
                            probability_matrix[batch_idx, pos] = self.key_mask_probability
                            key_positions_set.add(pos)
                    non_key_positions = all_positions - key_positions_set
                    for pos in non_key_positions:
                        probability_matrix[batch_idx, pos] = self.nonkey_mask_probability

        # Mask based on probabilities
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Compute loss only on masked tokens

        # Replace masked tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # Replace some masked tokens with random tokens
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, self.random_replace_prob)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def __call__(self, examples):
        """
        Batch and process examples.
        """
        # Stack input_ids, attention_mask, and other elements
        batch = {
            "input_ids": torch.stack([example["input_ids"] for example in examples]),
            "attention_mask": torch.stack([example["attention_mask"] for example in examples]),
            "key_positions": [example["key_positions"] for example in examples],
        }
        special_tokens_mask = torch.stack([example["special_tokens_mask"] for example in examples])

        # Mask tokens with key and non-key probabilities
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"],
            special_tokens_mask=special_tokens_mask,
            key_positions=batch["key_positions"],
        )
        return batch


def fasttext_detect_language(text):
    """Detect the language of a given text using FastText."""
    if not isinstance(text, str):
        text = str(text)
    try:
        lang, _ = model.predict(text)
        return lang[0].replace('__label__', '')
    except Exception:
        return 'unknown'
    
def is_table_english(rows, sample_size=20):
    """Check if a table is in English by sampling a few rows."""
    sample_text = ' '.join([str(row) for row in random.sample(rows, min(sample_size, len(rows)))])
    detected_lang = fasttext_detect_language(sample_text)
    return detected_lang == 'en'


def _hash_row(row):
    """Create a unique hash for a row to track duplicates."""
    filtered_row = {k: v for k, v in row.items() if k not in ["genre", "actor", "category", "brand", "filename"]}
    row_str = json.dumps(filtered_row)
    return hashlib.md5(row_str.encode()).hexdigest()

def create_data(path, path_is="dir", sample_num=None, pretraining_path=None):
    """
    Identical to `_load_data`, but outside the class. Processes and returns the dataset.
    
    Args:
        path (str): Path to the directory or file containing the data.
        path_is (str): Type of dataset processing. Options: 'dir', 'csv', 'json', 'test'.
        sample_num (int): Number of samples to extract from each table in the data directory.
        pretraining_path (str, optional): Path to pretraining data for filtering unseen test samples.

    Returns:
        List[dict]: Processed dataset.
    """
    data = []
    tables_skipped_randomly = 0
    max_random_skips = 10

    if path_is == "dir":
        for filename in os.listdir(path):
            if filename.endswith(".json"):
                filepath = os.path.join(path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f]  # Read all lines as JSON objects

                    if not is_table_english(rows):
                        print(f"Skipping non-English table: {filename}")
                        continue

                    if tables_skipped_randomly < max_random_skips and random.random() < 0.5:
                        tables_skipped_randomly += 1
                        print(f"Randomly skipping table: {filename} (Random Skip #{tables_skipped_randomly})")
                        continue

                    sampled_rows = random.sample(rows, min(sample_num, len(rows)))  # Sample rows
                    for row in sampled_rows:
                        for key in ["genre", "category", "filename"]:
                            row.pop(key, None)
                    data.extend(sampled_rows)

    elif path_is == "csv":
        if path is not None:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            table = pd.read_csv(StringIO(''.join(lines)))
            if "class" in table.columns:
                table.drop(columns=["class"], inplace=True)  # Drop 'class' column
            for _, row in table.iterrows():
                row_json = row.to_dict() 
                data.append(row_json)

    elif path_is == "json":
        if path is not None:
            with open(path, "r", encoding="utf-8") as json_file:
                for line in json_file:
                    json_line = json.loads(line)
                    data.append(json_line)

    elif path_is == "test":
        if pretraining_path is None:
            raise ValueError("Pretraining path must be provided to create the test set.")

        # Step 1: Load pretraining data and store hashed rows
        pretraining_hashes = set()
        with open(pretraining_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                pretraining_hashes.add(_hash_row(row))  # Store hashes of pretraining data

        # Step 2: Load test data and filter based on pretraining
        test_unseen = []

        json_files = sorted([f for f in os.listdir(path) if f.endswith(".json")])
        for filename in json_files:
            filepath = os.path.join(path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]

                if not is_table_english(rows):
                    print(f"Skipping non-English table: {filename}")
                    continue
                
                # Filter only unseen rows
                unseen_rows = [row for row in rows if _hash_row(row) not in pretraining_hashes]

                if len(unseen_rows) > sample_num:
                    sampled_rows = random.sample(unseen_rows, sample_num)
                else:
                    sampled_rows = unseen_rows
                
                test_unseen.extend(sampled_rows)

        data.extend(test_unseen)

    return data


class JSONDataset(Dataset):
    def __init__(self, path, tokenizer: PreTrainedTokenizer, max_length=512, path_is="json", version="jsonbert", sample_num=None, pretraining_path=None):
        """
        Args:
            path (str): Path to the directory or file containing the data.
            tokenizer (PreTrainedTokenizer): Tokenizer for processing the text.
            max_length (int): Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.path_is = path_is
        self.version = version
        self.sample_num = sample_num
        self.pretraining_path = pretraining_path

        self.data = create_data(path, path_is=path_is, sample_num=sample_num, pretraining_path=pretraining_path)

    # def _serialize(self, json_obj):
    #     token_ids = self.tokenizer.encode(str(json_obj), add_special_tokens=False)
    #     tokenized = self.tokenizer.convert_ids_to_tokens(token_ids)
    #     tokenized = [token for token in tokenized if token != "'"]
    #     return " ".join(tokenized)

    def _serialize(self, json_obj):
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

    
    # Ignore
    def _serialize_delimitter(self, json_obj, parent_key="", sep="."):
        """
        Serialize a JSON object into a string format suitable for tokenization, handling nested structures.
        """
        serialized = []
        for key, value in json_obj.items():
            full_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                serialized.append(self._serialize_delimitter(value, parent_key=full_key, sep=sep))
            elif isinstance(value, list):
                list_content = ", ".join([str(v) if not isinstance(v, dict) else self._serialize_delimitter(v, parent_key=full_key, sep=sep) for v in value])
                serialized.append(f"{full_key}: [{list_content}]")
                serialized.append("|")
            else:
                serialized.append(f"{full_key}: {value}")
                serialized.append("|")
        return " ".join(serialized)

    def _serialize_vanilla(self, json_obj, parent_key="", sep="."):
        """
        Serialize a JSON object into a string format suitable for tokenization, handling nested structures.
        """
        serialized = []
        for key, value in json_obj.items():
            full_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                serialized.append(self._serialize_vanilla(value, parent_key=full_key, sep=sep))
            elif isinstance(value, list):
                list_content = ", ".join([str(v) if not isinstance(v, dict) else self._serialize_vanilla(v, parent_key=full_key, sep=sep) for v in value])
                serialized.append(f"{full_key} is [{list_content}]")
                serialized.append(",")
            else:
                serialized.append(f"{full_key} is {value}")
                serialized.append(",")
        return " ".join(serialized)
    

    # def _find_key_positions(self, serialized, json_obj, parent_key=""):
    #     """
    #     Find token indices corresponding to keys in the tokenized serialized string, including handling of `|`.
    #     """
    #     tokenized = self.tokenizer(
    #         serialized,
    #         max_length=self.max_length,
    #         truncation=True,
    #         return_tensors="pt"
    #     )
    #     input_ids = tokenized["input_ids"].squeeze(0).tolist()
    #     tokenized_serialized = self.tokenizer.convert_ids_to_tokens(input_ids)

    #     key_positions = {}
    #     current_position = 1  # Start after [CLS]

    #     def recurse_json(obj, parent_key=""):
    #         """
    #         Recursively process JSON to find token positions.
    #         """
    #         nonlocal current_position
    #         if isinstance(obj, dict):
    #             if self.version != 'vanilla':
    #                 current_position += 1  # '{' token

    #             for key, value in obj.items():
    #                 full_key = f"{parent_key}.{key}" if parent_key else key
    #                 tokenized_key = self.tokenizer.tokenize(key)

    #                 key_start_pos = _find_token_indices(tokenized_serialized, tokenized_key, current_position)
    #                 if key_start_pos is not None:
    #                     key_positions[full_key] = key_start_pos

    #                 current_position = key_start_pos[-1] + 2 if key_start_pos else current_position + len(tokenized_key) + 2  # Account for `:` and space tokens

    #                 recurse_json(value, full_key)
    #                 current_position += 1  # `,` or `}` token

    #             if self.version != 'vanilla':
    #                 current_position += 1  # '}' token

    #         elif isinstance(obj, list):
    #             current_position += 1  # '[' token
    #             for i, element in enumerate(obj):
    #                 recurse_json(element, f"{parent_key}[{i}]")
    #             current_position += 1  # ']' token

    #         elif isinstance(obj, str):
    #             current_position += len(self.tokenizer.tokenize(obj))
    #         elif isinstance(obj, (int, float, bool, type(None))):
    #             current_position += len(self.tokenizer.tokenize(str(obj)))

    #     def _find_token_indices(sequence, tokens, start_index):
    #         """
    #         Find indices of tokens within sequence from start_index.
    #         """
    #         for i in range(start_index, len(sequence) - len(tokens) + 1):
    #             if sequence[i : i + len(tokens)] == tokens:
    #                 return list(range(i, i + len(tokens)))
    #         return None

    #     recurse_json(json_obj)
    #     return key_positions

    def _find_key_positions(self, serialized, json_obj, parent_key=""):
        tokenized = self.tokenizer(
            serialized,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0).tolist()
        tokenized_serialized = self.tokenizer.convert_ids_to_tokens(input_ids)

        key_positions = {}
        current_position = 1  # After [CLS]

        def recurse_json(obj, parent_key=""):
            nonlocal current_position
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{parent_key}.{key}" if parent_key else key
                    tokenized_key = self.tokenizer.tokenize(key)

                    key_start_pos = _find_token_indices(tokenized_serialized, tokenized_key, current_position)
                    if key_start_pos is not None:
                        key_positions[full_key] = key_start_pos
                    current_position = key_start_pos[-1] + 2 if key_start_pos else current_position + len(tokenized_key) + 2
                    recurse_json(value, full_key)
                    
            elif isinstance(obj, list):
                for i, element in enumerate(obj):
                    recurse_json(element, f"{parent_key}[{i}]")
                    
            else:
                current_position += len(self.tokenizer.tokenize(str(obj)))

        def _find_token_indices(sequence, tokens, start_index):
            for i in range(start_index, len(sequence) - len(tokens) + 1):
                if sequence[i:i+len(tokens)] == tokens:
                    return list(range(i, i+len(tokens)))
            return None

        recurse_json(json_obj)
        return key_positions

    def extract_keys_nonkeys(self, json_obj, parent_key=""):
        """
        Recursively extract keys and nonkeys from a JSON object for visualization.
        """
        if not isinstance(json_obj, dict):
            raise TypeError(f"Expected a dictionary, but got {type(json_obj)}: {json_obj}")
        keys = []
        nonkeys = []
        
        for key, value in json_obj.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                # Recursively handle nested dictionaries
                nested_keys, nested_nonkeys = self.extract_keys_nonkeys(value, parent_key=full_key)
                keys.extend(nested_keys)
                nonkeys.extend(nested_nonkeys)
            elif isinstance(value, list):
                # Handle lists
                for item in value:
                    if isinstance(item, dict):
                        nested_keys, nested_nonkeys = self.extract_keys_nonkeys(item, parent_key=f"{full_key}")
                        keys.extend(nested_keys)
                        nonkeys.extend(nested_nonkeys)
                    else:
                        keys.append(f"{full_key}")
                        nonkeys.append(str(item))
            else:
                # For scalar nonkeys
                keys.append(full_key)
                nonkeys.append(str(value))
        
        return keys, nonkeys

    def __getitem__(self, idx):
        """
        Return tokenized inputs and metadata for the collator.
        """
        json_obj = self.data[idx]

        if self.version == 'vanilla':
            serialized_vanilla = self._serialize_vanilla(json_obj)
            tokenized = self.tokenizer(
                serialized_vanilla, 
                max_length=self.max_length, 
                truncation=True, 
                padding="max_length",
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "token_type_ids": tokenized["token_type_ids"].squeeze(0),
                "special_tokens_mask": tokenized["special_tokens_mask"].squeeze(0),
            }

        serialized = self._serialize(json_obj)
        
        tokenized = self.tokenizer(
            serialized, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

        key_positions = self._find_key_positions(serialized, json_obj)
        
        if self.version == 'jsonbert':
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "token_type_ids": tokenized["token_type_ids"].squeeze(0),
                "special_tokens_mask": tokenized["special_tokens_mask"].squeeze(0),
                "key_positions": key_positions
            }
        elif self.version == 'bert':
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "token_type_ids": tokenized["token_type_ids"].squeeze(0),
                "special_tokens_mask": tokenized["special_tokens_mask"].squeeze(0),
            }
        elif self.version == 'visualize':
            keys, nonkeys = self.extract_keys_nonkeys(json_obj)
            return {
                "serialized": serialized,
                "keys": keys,
                "nonkeys": nonkeys,
                "key_positions": key_positions
            }

    def __len__(self):
        return len(self.data)
