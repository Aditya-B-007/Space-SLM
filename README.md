# Space LLM

This project is a decoder-only Transformer (GPT-like) model built from scratch in PyTorch. It features a custom 1-bit quantization for its linear layers to explore model compression techniques. The project also includes a complete Retrieval-Augmented Generation (RAG) demo that can scrape websites, build a vector index, and answer questions based on the scraped content.

## ✨ Features

-   **Decoder-Only Architecture**: Implements a GPT-style model for text generation.
-   **Built from Scratch**: All components, including Multi-Head Attention and Layer Normalization, are coded from the ground up.
-   **1-Bit Quantization**: Uses a custom `QuantizedLinear` layer that binarizes weights during the forward pass, reducing model size and computational cost.
-   **Text Generation**: Includes a flexible `generate_text` function with temperature and top-k sampling.
-   **RAG Demo**: A full-featured RAG application that demonstrates how to use the model for question-answering over custom documents.

## 📂 File Structure
.
├── quantized_gpt.py    # The complete model and application code
└── README.md           # You are here!
## ⚙️ Requirements

You'll need Python 3.8+ and the following libraries. You can install them using pip:

```bash
pip install torch tiktoken requests beautifulsoup4 faiss-cpu

1. Basic Text Generation

You can import the GPT_Model and helper functions into your own projects. Since the model in the script is untrained, the output will be random, but it demonstrates the generation pipeline.
Python

import torch
import tiktoken
from quantized_gpt import GPT_Model, GPT_CONFIG, text_to_token_ids, generate_text, token_ids_to_text

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize Model and Tokenizer
model = GPT_Model(GPT_CONFIG).to(device)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Optional: Load your pre-trained weights
# model.load_state_dict(torch.load("your_model_weights.pth"))

# 2. Prepare Input
start_context = "The future of AI is"
token_ids = text_to_token_ids(start_context, tokenizer).to(device)

# 3. Generate Text
generated_ids = generate_text(
    model=model,
    idx=token_ids,
    max_new_tokens=50,
    context_size=GPT_CONFIG["context_length"]
)

# 4. Decode the Output
generated_text = token_ids_to_text(generated_ids, tokenizer)
print(generated_text)

2. Run the RAG Application Demo

The quantized_gpt.py script can be run directly from the terminal to start the interactive RAG demo.

    Run the script:
    Bash

    python quantized_gpt.py

    The script will first show a simple generation example.

    Then, it will ask if you want to run the RAG demo. Type yes and press Enter.

    Provide URLs: Paste 1-3 website URLs you want to use as a knowledge base, separated by spaces.

    Ask Questions: Once the index is built, you can ask questions about the content of the websites. The model will retrieve relevant chunks of text and use them to generate an answer. Type exit to quit.

🔧 Model Configuration

The model's hyperparameters are defined in the GPT_CONFIG dictionary at the top of quantized_gpt.py.

    vocab_size: The number of unique tokens in the vocabulary (defaults to cl100k_base).

    context_length: The maximum number of tokens the model can process at once.

    embedding_dim: The size of the token and position embedding vectors.

    num_heads: The number of heads in the Multi-Head Attention mechanism.

    n_layers: The number of Transformer blocks to stack.

    dropout: The dropout rate for regularization.

    qkv_bias: Whether to use a bias in the query, key, and value projection layers.

📄 License

This project is open-source and available under the MIT License.
