import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertConfig
from safetensors.torch import load_file


class JSONBERT_NEWLOSS_1(BertForMaskedLM):
    def __init__(self, config, tokenizer, model_path=None, lambda_align=0.4):
        super(JSONBERT_NEWLOSS_1, self).__init__(config)
        self.tokenizer = tokenizer

        # Custom embedding for keys
        self.key_embedding = nn.Embedding(self.tokenizer.vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Debug #
        if self.key_embedding.weight.requires_grad:
            print("Key embeddings are trainable!")

        # Load pre-trained weights if provided
        if model_path:
            state_dict = load_file(os.path.join(model_path, "model.safetensors"))
            self.load_state_dict(state_dict, strict=False)
            print(f"Pre-trained JSONBERT_NEWLOSS loaded from {model_path}")
        else:
            pretrained_bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
            self.bert = pretrained_bert.bert
            self.cls = pretrained_bert.cls
            self.key_embedding.weight.data = (
                pretrained_bert.bert.embeddings.word_embeddings.weight.data.clone()
            )

        # Dictionary to store contextual embeddings for centroid loss
        self.key_to_contextual_embeddings = {}

        # Loss weight
        self.lambda_align = lambda_align

    def _replace_key_embeddings(self, input_ids, key_positions, sequence_output):
        """
        Replace embeddings at key token positions with custom key embeddings.
        Collect contextualized embeddings for centroid loss computation.
        """
        batch_indices, token_indices = [], []
        for batch_idx, key_dict in enumerate(key_positions):
            if not key_dict:
                print(f"Warning: Empty key_positions for batch index {batch_idx}")

            for full_key, positions in key_dict.items():
                sub_tokens = self.tokenizer.tokenize(full_key)
                token_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)

                for token_id, pos in zip(token_ids, positions):
                    if token_id not in self.key_to_contextual_embeddings:
                        self.key_to_contextual_embeddings[token_id] = []

                    # Detach embeddings before storing to avoid memory issues
                    self.key_to_contextual_embeddings[token_id].append(sequence_output[batch_idx, pos].detach().unsqueeze(0))

                    batch_indices.append(batch_idx)
                    token_indices.append(pos)

    def compute_centroid_alignment_loss(self):
        """
        Compute the centroid alignment loss using cosine similarity.
        """
        if not self.key_to_contextual_embeddings:
            return torch.tensor(0.0, device=self.key_embedding.weight.device)  # Avoid division by zero

        total_loss = torch.tensor(0.0, device=self.key_embedding.weight.device, requires_grad=True)
        cnt = 0

        for key_token_id, contextual_embeddings in self.key_to_contextual_embeddings.items():
            contextual_embeddings = torch.cat(contextual_embeddings, dim=0)
            if len(contextual_embeddings) > 1:  # Compute only if multiple contexts exist
                centroid = contextual_embeddings.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            else:
                centroid = contextual_embeddings

                # Retrieve key embedding without creating a new tensor
                key_embedding = self.layer_norm(self.key_embedding.weight[key_token_id].unsqueeze(0))

                # Compute cosine similarity loss
                similarity = F.cosine_similarity(centroid, key_embedding, dim=-1)
                loss = 1 - similarity.mean()

                total_loss = total_loss + loss
                cnt += 1

        # Average loss
        centroid_alignment_loss = total_loss / cnt if cnt > 0 else total_loss

        self.key_to_contextual_embeddings = {}

        return centroid_alignment_loss

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, key_positions=None, compute_alignment_loss=False, **kwargs):
        device = input_ids.device
        attention_mask = attention_mask.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        if labels is not None:
            labels = labels.to(device)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state

        if key_positions is not None:
            self._replace_key_embeddings(input_ids, key_positions, sequence_output)

        prediction_scores = self.cls(sequence_output)

        loss = None
        mlm_loss = torch.tensor(0.0, device=device)
        centroid_alignment_loss = torch.tensor(0.0, device=device)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if compute_alignment_loss:
            centroid_alignment_loss = self.compute_centroid_alignment_loss()

        loss = mlm_loss + self.lambda_align * centroid_alignment_loss

        return {
            "loss": loss, 
            "mlm_loss": mlm_loss,
            "centroid_alignment_loss": centroid_alignment_loss,
            "logits": prediction_scores, 
            "hidden_states": sequence_output
        }