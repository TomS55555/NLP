import torch
import torch.nn as nn
from torch.functional import F


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim) -> None:
        super().__init__()
        self.E = torch.randn(num_embeddings, embedding_dim)
    def forward(self, x):
        return self.E[x]

class Attention(nn.Module):
    def __init__(self, emb_dim, context_length, query_dim) -> None:
        super().__init__()
        self.query_dim = query_dim 
        self.emb_dim = emb_dim
        self.Q = nn.Linear(emb_dim, query_dim)
        self.K = nn.Linear(emb_dim, query_dim)
        self.V = nn.Linear(emb_dim, query_dim)
        self.register_buffer('tril_mask', torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, x):
        B, C, E = x.shape # x is a batch of encoded inputs: (B, C, E)
        queries = self.Q(x)  # (B, C, query_dim)
        keys = self.K(x)  # (B, C, query_dim)
        values = self.V(x) # (B, C, query_dim)
        activations = queries @ keys.transpose(-2, -1) * self.query_dim**-0.5 # (B, C, C)
        activations = activations.masked_fill(self.tril_mask[:C, :C] == 0, float('-inf'))
        weights = F.softmax(activations, dim=-1)  # (B, C, C)
        outputs = weights @ values # (B, context_length, query_dim)
        return outputs

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention is parallel """
    def __init__(self, num_heads, head_size, context_length) -> None:
        super().__init__()
        emb_dim = num_heads * head_size
        self.heads = nn.ModuleList([Attention(emb_dim=emb_dim, context_length=context_length, query_dim=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, num_heads*head_size)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        x = self.net(x)
        return x 


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim, n_heads, context_length) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0
        head_size = emb_dim // n_heads
        self.sa = MultiHeadAttention(num_heads=n_heads, head_size=head_size, context_length=context_length)
        self.ffwd = MLP(emb_dim, 4*emb_dim, emb_dim)
        self.l1 = nn.LayerNorm(emb_dim)
        self.l2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.sa(self.l1(x)) # apply layer normalization before attention
        x = x + self.ffwd(self.l2(x)) # apply layer normalization once more before feed forward
        return x


