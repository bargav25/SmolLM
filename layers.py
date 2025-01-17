import torch
from torch import nn
from attention import RopeAttention

# Multi-Layer Perceptron (MLP) class
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        """
        Initialize an MLP with a gating mechanism and non-linear activation.

        Args:
            hidden_size (int): The size of the input and output embeddings.
            intermediate_size (int): The size of the hidden layer.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Linear layers for the gated MLP structure
        self.W_gate = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.W_up = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.W_down = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Activation function (SiLU)
        self.act_fn = torch.nn.modules.activation.SiLU()

    def forward(self, x):
        """
        Forward pass for the gated MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Apply gating mechanism and project back to hidden size
        down_proj = self.W_down(self.act_fn(self.W_gate(x)) * self.W_up(x))
        return down_proj


# RMSNorm class (Root Mean Square Normalization)
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Initialize RMSNorm.

        Args:
            hidden_size (int): The size of the input embeddings.
            eps (float): A small value to prevent division by zero.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Learnable scaling factor
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Forward pass for RMSNorm.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Calculate variance along the last dimension (hidden size)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        # Normalize and scale
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


# Rotary Embedder class
class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        """
        Initialize rotary embeddings for positional encodings.

        Args:
            dim (int): Dimensionality of embeddings (half the hidden size per head).
            base (float): Base frequency for rotary embeddings.
        """
        super().__init__()
        # Frequency for rotary embeddings
        self.freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def forward(self, x):
        """
        Compute cosine and sine embeddings for rotary position encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine embeddings.
        """
        pos = torch.arange(x.shape[-2], dtype=torch.float32)  # Sequence positions

        # Calculate angular frequencies
        angles = torch.einsum("p,f->pf", pos, self.freq).unsqueeze(0)
        
        # Create cosine and sine embeddings
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos(), emb.sin()


# LlamaDecoder class
class LlamaDecoder(nn.Module):
    def __init__(self, config):
        """
        Initialize the LlamaDecoder layer with attention and MLP sublayers.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        # Attention mechanism with rotary embeddings
        self.self_attn = RopeAttention(config)

        # Feedforward neural network (MLP)
        self.mlp = MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size)

        # Pre-layer normalization (RMSNorm) for attention and MLP
        self.pre_attn_rmsnorm = RMSNorm(config.hidden_size, eps=1e-05)
        self.pre_mlp_rmsnorm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, hidden_states, attention_mask):
        """
        Forward pass for the LlamaDecoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (torch.Tensor): Mask to prevent attention to certain positions.

        Returns:
            Tuple[torch.Tensor]: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Residual connection for attention sublayer
        residual = hidden_states

        # Apply RMSNorm before attention
        hidden_states = self.pre_attn_rmsnorm(hidden_states)

        # Generate a triangular attention mask (causal masking)
        attention_mask = torch.triu(torch.full((attention_mask.shape[-1], attention_mask.shape[-1]),
                                               fill_value=float('-inf')), diagonal=1)

        # Apply self-attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        # Add residual connection
        hidden_states += residual

        # Residual connection for MLP sublayer
        residual = hidden_states

        # Apply RMSNorm before MLP
        hidden_states = self.pre_mlp_rmsnorm(hidden_states)

        # Pass through MLP
        hidden_states = self.mlp(hidden_states)

        # Add residual connection
        hidden_states += residual

        # Return the output hidden states
        return hidden_states,