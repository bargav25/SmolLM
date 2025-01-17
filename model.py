import torch
from torch import nn
from layers import MLP, RMSNorm, LlamaDecoder, RotaryEmbedder  # Import components from layers
from attention import RopeAttention  # Import attention mechanism

# The main model containing the embedding layer, decoder stack, and normalization layer
class smolModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Token embedding layer: maps input tokens to dense representations
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
        )
        # Stack of decoder layers (LlamaDecoder) defined by the configuration
        self.layers = nn.ModuleList([
            LlamaDecoder(config) for _ in range(config.num_hidden_layers)
        ])
        # RMSNorm: final layer normalization applied to hidden states
        self.norm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, input_ids=None, attention_mask=None):
        # Convert input token IDs to dense embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # Pass embeddings through each decoder layer in the stack
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,  # Pass the attention mask
            )
            # Update hidden states with the output of the current decoder layer
            hidden_states = layer_outputs[0]

        # Apply final layer normalization to the hidden states
        hidden_states = self.norm(hidden_states)

        # Return the processed hidden states
        return hidden_states


# The complete language model, combining smolModel and a language modeling head (lm_head)
class smolLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Core model containing embeddings and decoder stack
        self.model = smolModel(config)
        # Language modeling head: projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights between the embedding layer and lm_head
        self.tie_weights()

    def tie_weights(self):
        # Ensures the lm_head shares weights with the embedding layer
        # This is a common optimization for language models
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask):
        # Pass inputs through the core model to obtain hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Obtain hidden states from the model's output
        hidden_states = outputs

        # Pass hidden states through the language modeling head
        logits = self.lm_head(hidden_states)
        # Ensure logits are returned in float for numerical stability
        logits = logits.float()

        # Return the output as a dictionary containing logits
        return {'logits': logits}