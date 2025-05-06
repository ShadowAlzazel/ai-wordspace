import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from scipy.stats import binom
from dotenv import load_dotenv

# Suppress HF logs for brevity
from transformers import logging
logging.set_verbosity_error()

# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

# 1) Load tokenizer and model (only embeddings)
def load_embeddings(model_name, token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=token,
        padding_side="right"
    )
    model = AutoModel.from_pretrained(
        model_name,
        use_auth_token=token,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model.get_input_embeddings()

# 2) Prepare reference embedding
print("Loading tokenizer and embedding layer...")
tokenizer, embedding_layer = load_embeddings(MODEL_NAME, HUGGINGFACE_TOKEN)
vocab_size, D = embedding_layer.weight.shape
ref_token = "tea"
ref_id = tokenizer.convert_tokens_to_ids(ref_token)
ref_vec = F.normalize(embedding_layer.weight[ref_id], dim=0)

# 3) Experiment parameters
tau = 0.5          # similarity threshold
n = 100            # tokens per trial
trials = 500       # number of trials
counts = []

print(f"Running {trials} trials of {n} samples each...")
for _ in range(trials):
    sample_ids = random.sample(range(vocab_size), n)
    # compute cosine similarities in batch
    batch = F.normalize(embedding_layer.weight[sample_ids], dim=1)
    sims = (batch @ ref_vec).cpu().detach().numpy()
    counts.append((sims > tau).sum())

# 4) Fit Binomial and plot results
p_hat = np.mean([c / n for c in counts])
values, freqs = np.unique(counts, return_counts=True)
freqs = freqs / trials

plt.figure(figsize=(8, 5))
plt.bar(values, freqs, alpha=0.6, label='Empirical counts')
plt.plot(
    values,
    binom.pmf(values, n, p_hat),
    'o-',
    label=f'Binomial(n={n}, p̂={p_hat:.3f})'
)
plt.xlabel('Number of successes per trial')
plt.ylabel('Probability')
plt.title('Token-Similarity Success Counts vs. Binomial PMF')
plt.legend()
plt.tight_layout()
plt.show()

# 5) Print summary
print(f"Estimated p̂ = {p_hat:.4f} (threshold τ={tau})")