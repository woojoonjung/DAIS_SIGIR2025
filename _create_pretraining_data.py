import os
import json
from dataset import create_data

def create_pretraining_data(data_path, pretraining_save_path, sample_num=450):
    if os.path.exists(pretraining_save_path):
        print(f"✅ Pretraining dataset already exists: {pretraining_save_path}")
        return

    print(f"🔄 Creating pretraining dataset from {data_path}...")

    # Load processed dataset
    data = create_data(data_path, path_is="dir", sample_num=sample_num)

    # Save pretraining data
    os.makedirs(os.path.dirname(pretraining_save_path), exist_ok=True)
    with open(pretraining_save_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    print(f"✅ Pretraining dataset saved at: {pretraining_save_path}")

if __name__ == "__main__":
    DATA_PATH = "./data/Product_top100"
    PRETRAINING_SAVE_PATH = "./data/pretraining_data_product.jsonl"
    
    create_pretraining_data(DATA_PATH, PRETRAINING_SAVE_PATH)
