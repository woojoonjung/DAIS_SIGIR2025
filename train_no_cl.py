import os
import argparse
import torch
import random
import json
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import AdamW, BertConfig, get_scheduler
from transformers.utils.logging import set_verbosity_info, enable_propagation
from dataset import JSONDataset, JSONDataCollator
from no_cl import JSONBERT_INTERPOLATE
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

SEED = 42  
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Train JSONBERT with Masked Language Modeling")

    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")

    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased", help="Pretrained model name or path")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=9, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")

    parser.add_argument("--logging_steps", type=int, default=500, help="Steps interval for logging")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="TensorBoard logging directory")

    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Probability for masking tokens in MLM")
    parser.add_argument("--key_mask_probability", type=float, default=0.24, help="Probability for masking keys")
    parser.add_argument("--nonkey_mask_probability", type=float, default=0.17, help="Probability for masking nonkeys")

    return parser.parse_args()


def train_model(args):
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available and will be used.")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU.")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    # Load dataset
    dataset = JSONDataset(
        path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    # Data collator
    data_collator = JSONDataCollator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        key_mask_probability=args.key_mask_probability,
        nonkey_mask_probability=args.nonkey_mask_probability,
        hybrid_epochs=6, 
        total_epochs=args.num_epochs,
    )

    # Data loaders
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    # Initialize model
    config = BertConfig.from_pretrained(args.pretrained_model)
    model = JSONBERT_INTERPOLATE(config, tokenizer)
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    # TensorBoard for logging
    writer = SummaryWriter(log_dir=args.logging_dir)

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(num_training_steps), desc="Training")

    for epoch in range(args.num_epochs):
        # Set the current epoch in the data collator
        data_collator.set_epoch(epoch)

        # Training phase
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            key_positions = batch["key_positions"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                key_positions=key_positions,
            )
            loss = outputs["loss"]
            loss = loss / args.gradient_accumulation_steps  # Scale loss for gradient accumulation
            loss.backward()
            total_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)

            # Log train loss
            if global_step % args.logging_steps == 0:
                writer.add_scalar("train_loss", loss.item() * args.gradient_accumulation_steps, global_step)

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{args.num_epochs} - Train Loss: {avg_train_loss:.4f}")

    # Save model checkpoint
    epoch_save_path = os.path.join(args.output_dir, f"epoch-{epoch + 1}")
    os.makedirs(epoch_save_path, exist_ok=True)
    model.save_pretrained(epoch_save_path)
    tokenizer.save_pretrained(epoch_save_path)
    print(f"Model saved after epoch {epoch + 1} at {epoch_save_path}")

    writer.close()



if __name__ == "__main__":
    args = parse_args()

    train_model(args)