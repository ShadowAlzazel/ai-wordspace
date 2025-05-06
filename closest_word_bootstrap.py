from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

def get_closest_words(vector, embedding_layer, tokenizer, top_n=10, epsilon=1e-8):
    """Find closest words to a given vector using cosine similarity"""
    # Normalize the input vector and embeddings
    vector = F.normalize(vector.unsqueeze(0), p=2, dim=1, eps=epsilon)
    embeddings = F.normalize(embedding_layer.weight, p=2, dim=1, eps=epsilon)
    
    # Compute cosine similarity
    similarity = torch.matmul(vector, embeddings.T).squeeze(0)
    
    # Get the top N most similar words
    top_values, top_indices = torch.topk(similarity, top_n)
    top_words = [tokenizer.decode([idx.item()]).strip() for idx in top_indices]
    top_scores = [similarity[idx].item() for idx in top_indices]
    
    return list(zip(top_words, top_scores))

def main():
    print(f"Loading model: {MODEL_NAME}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=HUGGINGFACE_TOKEN,
        padding_side="right"
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        use_auth_token=HUGGINGFACE_TOKEN,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
    # Get embedding layer
    embedding_layer = model.get_input_embeddings()
    vocab_size, embedding_dim = embedding_layer.weight.shape
    print(f"Embedding layer: {vocab_size} tokens × {embedding_dim} dimensions")
    
    # Example from paper: tea + milk
    word1 = "tea"
    word2 = "milk"
    
    # Get token IDs
    word1_ids = tokenizer.encode(word1, add_special_tokens=False)
    word2_ids = tokenizer.encode(word2, add_special_tokens=False)
    
    if not word1_ids or not word2_ids:
        print(f"One or both words not found in vocabulary: {word1}, {word2}")
        return
        
    # Get embeddings
    word1_vector = embedding_layer(torch.tensor([word1_ids[0]]))
    word2_vector = embedding_layer(torch.tensor([word2_ids[0]]))
    
    # Create composite vector
    composite_vector = word1_vector + word2_vector
    
    # Store results from bootstrap trials
    T = 100  # Number of bootstrap trials
    semantic_set = {"tea", "milk", "boba", "chai", "latte", "coffee"}
    semantic_ranks = {word: [] for word in semantic_set}
    semantic_sims = {word: [] for word in semantic_set}
    
    # Run bootstrap trials
    print(f"Running {T} bootstrap trials with noise...")
    for t in tqdm(range(T)):
        # Add small Gaussian noise
        noise = torch.randn_like(composite_vector) * 0.01
        noisy_vector = composite_vector + noise
        
        # Get closest words
        closest = get_closest_words(noisy_vector.squeeze(0), embedding_layer, tokenizer, top_n=20)
        
        # Record ranks and similarities for semantic set
        words_dict = {word: (rank, sim) for rank, (word, sim) in enumerate(closest)}
        
        # Check each word in semantic set
        for word in semantic_set:
            for result_word, (rank, sim) in words_dict.items():
                if word in result_word:  # Partial match to handle tokenization differences
                    semantic_ranks[word].append(rank + 1)  # 1-indexed rank
                    semantic_sims[word].append(sim)
                    break
    
    # Plot similarity distributions for selected words
    plt.figure(figsize=(12, 8))
    
    # Plot histogram for "boba" and "chai" similarities
    for word, color in [("boba", "blue"), ("chai", "green")]:
        if semantic_sims[word]:
            sims = semantic_sims[word]
            mean_sim = np.mean(sims)
            std_sim = np.std(sims)
            
            plt.hist(sims, bins=15, alpha=0.4, color=color, 
                     label=f"{word}: μ={mean_sim:.2f}, σ={std_sim:.2f}")
            
            # Overlay fitted Gaussian
            x = np.linspace(min(sims)-0.05, max(sims)+0.05, 100)
            plt.plot(x, norm.pdf(x, mean_sim, std_sim) * len(sims) * (max(sims)-min(sims))/15,
                     color=color, linewidth=2)
    
    # Background distribution for reference
    x_bg = np.linspace(-0.1, 0.3, 100)
    bg_mean, bg_std = 0, 1/np.sqrt(embedding_dim)
    plt.plot(x_bg, norm.pdf(x_bg, bg_mean, bg_std) * 10,
             'r--', label=f"Background: N(0, 1/√D)")
    
    plt.title(f"'tea + milk' Similarity Distribution over {T} Bootstrap Trials")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Print statistical summary
    print("\nStatistical Summary:")
    print(f"{'Word':<10} {'Mean Rank':<12} {'Rank Var':<12} {'Mean Sim':<12} {'Sim Var':<12}")
    print("-" * 60)
    
    for word in semantic_set:
        if semantic_ranks[word]:
            mean_rank = np.mean(semantic_ranks[word])
            var_rank = np.var(semantic_ranks[word])
            mean_sim = np.mean(semantic_sims[word])
            var_sim = np.var(semantic_sims[word])
            print(f"{word:<10} {mean_rank:<12.2f} {var_rank:<12.2f} {mean_sim:<12.4f} {var_sim:<12.6f}")
    
    # Show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
