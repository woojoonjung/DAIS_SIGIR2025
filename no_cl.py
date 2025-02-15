import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertConfig
from safetensors.torch import load_file


class JSONBERT_INTERPOLATE(BertForMaskedLM):
    def __init__(self, config, tokenizer, model_path=None):
        """
        Args:
            config (BertConfig): Configuration for the BERT model.
            tokenizer: Tokenizer for the model.
            model_path (str, optional): Path to pre-trained model weights.
        """
        super(JSONBERT_INTERPOLATE, self).__init__(config)
        self.tokenizer = tokenizer

        # Custom layers for keys
        self.key_embedding = nn.Embedding(self.tokenizer.vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Learnable blending coefficient
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Start with equal weighting

        # Debug #
        if self.key_embedding.weight.requires_grad == True:
            print("Key embeddings are trainable!")

        # Load pre-trained weights if provided
        if model_path:
            state_dict = load_file(os.path.join(model_path, "model.safetensors"))
            self.load_state_dict(state_dict, strict=False)
            print(f"Pre-trained JSONBERT_INTERPOLATE loaded from {model_path}")
        else:
            # Load base BERT weights
            pretrained_bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
            self.bert = pretrained_bert.bert
            self.cls = pretrained_bert.cls
            # Clone word embeddings to initialize key_embedding
            self.key_embedding.weight.data = (
                pretrained_bert.bert.embeddings.word_embeddings.weight.data.clone()
            )

    def _replace_key_embeddings(self, input_ids, key_positions, sequence_output):
        """
        Replace embeddings at key token positions with custom key embeddings.
        """
        batch_indices, token_indices = [], []

        for batch_idx, key_dict in enumerate(key_positions):
            if not key_dict:
                print(f"Warning: Empty key_positions for batch index {batch_idx}")
            for positions in key_dict.values():
                batch_indices.extend([batch_idx] * len(positions))
                token_indices.extend(positions)

        # Efficient tensor operations for embedding replacement
        if batch_indices and token_indices:
            batch_indices = torch.tensor(batch_indices, device=sequence_output.device)
            token_indices = torch.tensor(token_indices, device=sequence_output.device)
            token_ids = input_ids[batch_indices, token_indices]

            key_embeddings = self.layer_norm(self.key_embedding(token_ids))
            # Debug #
            # original_embeddings = sequence_output[batch_indices, token_indices]
            # print(f"Cosine sim between key & bert: {F.cosine_similarity(original_embeddings, key_embeddings, dim=-1)}")
            # print(f"Cosine sim between key_normed & bert: {F.cosine_similarity(original_embeddings, key_normed, dim=-1)}")
            # print(f"Cosine sim between key_transformed & bert: {F.cosine_similarity(original_embeddings, key_transformed, dim=-1)}")
            #########
            sequence_output[batch_indices, token_indices] = (
                self.alpha * sequence_output[batch_indices, token_indices] +
                (1 - self.alpha) * key_embeddings
            )


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, key_positions=None, **kwargs):
        # Move inputs to device
        device = input_ids.device
        attention_mask = attention_mask.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Forward pass through BERT encoder
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state

        # Replace key token embeddings
        if key_positions is not None:
            self._replace_key_embeddings(input_ids, key_positions, sequence_output)

        # Compute prediction scores and optional loss
        prediction_scores = self.cls(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return {
            "loss": loss, 
            "logits": prediction_scores, 
            "hidden_states": sequence_output
        }