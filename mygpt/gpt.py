from mynn import *
import torch 

context_length = 20
batch_size = 64
emb_dim = 20
n_heads = 4
n_layer = 4

max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('input.txt', 'r', encoding='utf-8') as f:
    text_input = f.read()
chars = sorted(list(set(text_input)))
vocab_size = len(chars)
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda enc : ''.join([itos[i] for i in enc])

data = torch.tensor(encode(text_input), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data)-context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class GPTlanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=emb_dim)
        self.blocks = nn.Sequential(*[TransformerEncoderLayer(emb_dim=emb_dim, n_heads=n_heads, context_length=context_length) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        B, C = x.shape 
        tok_emb = self.token_embeding(x)
        pos_emb = self.pos_embedding(torch.arange(C, device=device))
        x = tok_emb + pos_emb  # (B, C, emb_dim)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None 
        else:    
            B, C, E = logits.shape 
            logits = logits.view(B*C, E)
            targets = targets.view(B*C)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, x, max_new_tokens):
        # x is (B, C) array of indices in current context
        for _ in range(max_new_tokens):
            x_cond = x[:, -context_length:]
            logits, _ = self(x_cond)  # Logits has size (B, C, vocab_size)
            logits = logits[:, -1, :]  # focus only on last character
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=-1)
        return x 

model = GPTlanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

torch.manual_seed(42)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))     

