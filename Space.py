# quantized_gpt.py

import torch
import torch.nn as nn
import math
import tiktoken
import requests
from bs4 import BeautifulSoup
import numpy as np
from faiss import IndexFlatL2
# --- Configuration ---
GPT_CONFIG = {
    'vocab_size': 100277, 
    'context_length': 256,   
    'embedding_dim': 512, 
    'num_heads': 16,        
    'n_layers': 12,        
    'dropout': 0.1,         
    'qkv_bias': False          
}


# --- 1. Model Components ---

class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        mask = (input.abs() <= 1).float()
        grad_input = grad_output * mask
        return grad_input

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Binarize the weights during the forward pass
        binarized_weight = BinarizeFunction.apply(self.weight)
        return torch.nn.functional.linear(input, binarized_weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], 4 * config["embedding_dim"]),
            Swish(),
            nn.Linear(4 * config["embedding_dim"], config["embedding_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class LayerNormalization(nn.Module):
    """Custom Layer Normalization module."""
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class MultiHeadAttention(nn.Module):
    """Quantized Multi-Head Causal Attention module."""
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Use the custom QuantizedLinear layer
        self.W_query = QuantizedLinear(d_in, d_out, bias=qkv_bias)
        self.W_key = QuantizedLinear(d_in, d_out, bias=qkv_bias)
        self.W_value = QuantizedLinear(d_in, d_out, bias=qkv_bias)
        self.out_proj = QuantizedLinear(d_out, d_out, bias=True) # Output projection usually has bias
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(2, 3)
        
        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec

class TransformerBlock(nn.Module):
    """A single Transformer block combining multi-head attention and feed-forward network."""
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config["embedding_dim"],
            d_out=config["embedding_dim"],
            context_length=config['context_length'],
            dropout=config['dropout'],
            num_heads=config['num_heads'],
            qkv_bias=config['qkv_bias']
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNormalization(config["embedding_dim"])
        self.norm2 = LayerNormalization(config["embedding_dim"])
        self.drop_resid = nn.Dropout(config['dropout'])

    def forward(self, x):
        # Attention with pre-normalization and residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut
        
        # Feed-forward with pre-normalization and residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        
        return x


# --- 2. Main GPT Model ---

class GPT_Model(nn.Module):
    """The main GPT model architecture."""
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNormalization(cfg["embedding_dim"])
        self.out_head = QuantizedLinear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeddings = self.tok_emb(in_idx.long())
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=in_idx.device)
        positional_embeddings = self.pos_emb(positions)
        
        x = token_embeddings + positional_embeddings
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits


# --- 3. Utility Functions ---

def generate_text(model, idx, max_new_tokens, context_size, temperature=0.4, top_k=3):
    model.eval() # Set model to evaluation mode
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        # Optional top-k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val.unsqueeze(-1), torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply temperature
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        # Append the new token
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)


# --- 4. RAG Application (Executed when the script is run directly) ---

def scrape_websites(urls: list[str]) -> list[str]:
    """Scrapes text content from a list of URLs."""
    documents = []
    print("Scraping website content...")
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\\n", strip=True)
            documents.append(text)
            print(f"  - Successfully scraped {url}")
        except requests.RequestException as e:
            print(f"  - Error fetching {url}: {e}")
    return documents

def split_text(texts: list[str], chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Splits long texts into smaller, overlapping chunks."""
    all_chunks = []
    for text in texts:
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        all_chunks.extend(chunks)
    return all_chunks

def get_embedding(text: str, model, tokenizer, device, context_length) -> list[float]:
    """Generates a sentence embedding for a given text."""
    token_ids = text_to_token_ids(text, tokenizer)
    token_ids = token_ids.to(device)
    if token_ids.shape[1] > context_length:
        token_ids = token_ids[:, :context_length]
        
    with torch.no_grad():
        token_embeddings = model.tok_emb(token_ids)
        sentence_embedding = torch.mean(token_embeddings.squeeze(0), dim=0)
        
    return sentence_embedding.cpu().numpy().tolist()

def build_faiss_index(text_chunks: list[str], model, tokenizer, device, context_length) -> tuple:
    if 'IndexFlatL2' not in globals():
        raise ImportError("FAISS is not installed. Cannot build index.")
        
    print("Generating embeddings for text chunks...")
    embeddings = [get_embedding(chunk, model, tokenizer, device, context_length) for chunk in text_chunks]
    
    dim = len(embeddings[0])
    index = IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    
    return index, text_chunks

def search_faiss(query_embedding: list[float], index, top_k: int = 3) -> list[int]:
    if 'IndexFlatL2' not in globals():
        raise ImportError("FAISS is not installed. Cannot perform search.")
        
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return I[0]


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = GPT_Model(GPT_CONFIG).to(device)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # ---Simple Text Generation ---
    print("\n--- Simple Text Generation Demo ---")
    start_context = "The capital of India is"
    token_ids = text_to_token_ids(start_context, tokenizer).to(device)

    print(f"Prompt: '{start_context}'")
    
    # Generate text (note: the model is untrained, so output will be random)
    generated_ids = generate_text(
        model=model,
        idx=token_ids,
        max_new_tokens=15,
        context_size=GPT_CONFIG["context_length"],
        temperature=0.7,
        top_k=10
    )
    generated_text = token_ids_to_text(generated_ids, tokenizer)
    print(f"Generated Text: {generated_text}\n")

    # --- RAG Application ---
    print("\n--- Retrieval-Augmented Generation (RAG) Demo ---")
    run_rag = input("Do you want to run the RAG demo? (yes/no): ").strip().lower()

    if run_rag == 'yes':
        if 'IndexFlatL2' not in globals():
            print("\nCannot run RAG demo because 'faiss-cpu' is not installed. Please run 'pip install faiss-cpu'.")
        else:
            urls = input("Enter 1-3 website URLs to scrape, separated by spaces: ").split()
            if not urls:
                print("No URLs provided. Skipping RAG demo.")
            else:
                raw_docs = scrape_websites(urls)
                text_chunks = split_text(raw_docs)
                print(f"Created {len(text_chunks)} text chunks.")

                index, chunks_map = build_faiss_index(
                    text_chunks, model, tokenizer, device, GPT_CONFIG["context_length"]
                )
                print("FAISS index built successfully.")

                while True:
                    query = input("\nAsk a question about the content (or type 'exit'): ").strip()
                    if query.lower() == 'exit':
                        break
                    
                    # 1. Retrieve relevant context
                    query_emb = get_embedding(query, model, tokenizer, device, GPT_CONFIG["context_length"])
                    top_indices = search_faiss(query_emb, index)
                    relevant_context = "\\n---\\n".join([chunks_map[i] for i in top_indices])
                    
                    # 2. Generate answer with context
                    final_prompt = f"Based on this context:\\n{relevant_context}\\n\\nAnswer this question: {query}"
                    
                    print("\nGenerating answer...")
                    prompt_token_ids = text_to_token_ids(final_prompt, tokenizer).to(device)
                    
                    output_ids = generate_text(
                        model=model,
                        idx=prompt_token_ids,
                        max_new_tokens=100,
                        context_size=GPT_CONFIG["context_length"]
                    )
                    
                    answer = token_ids_to_text(output_ids, tokenizer)
                    # Clean up the answer by removing the original prompt
                    final_answer = answer.replace(final_prompt, "").strip()
                    print(f"\\nAnswer:\\n{final_answer}")