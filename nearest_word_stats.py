#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-v0.3"

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze distribution of top-k similarities in embedding space')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_NAME, help='Model name or path')
    parser.add_argument('--num_trials', type=int, default=1000, help='Number of simulation trials')
    parser.add_argument('--embed_dim', type=int, default=4096, help='Embedding dimension for simulations')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size for simulations')
    parser.add_argument('--k', type=int, default=5, help='Number of top similarities to analyze')
    parser.add_argument('--use_real_model', action='store_true', help='Use real model instead of simulation')
    parser.add_argument('--query_words', type=str, nargs='+', default=['computer', 'ocean', 'democracy', 'happiness'],
                       help='Query words when using real model')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    return parser.parse_args()

def get_embeddings(model, tokenizer, text):
    """Get embeddings for input text using the model"""
    # Handle empty strings or None values
    if not text:
        return None
    
    # Filter out empty strings to avoid tokenization errors
    if isinstance(text, list):
        text = [t for t in text if t and isinstance(t, str)]
        if not text:
            return None
    
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the last hidden state as embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        print(f"Problematic text: {text[:100] if isinstance(text, str) else [t[:50] for t in text[:3]]}")
        return None

def simulate_top_k_similarities(embed_dim, vocab_size, k, num_trials):
    """
    Simulate the distribution of top-k similarities between a query vector
    and a set of vocabulary vectors.
    
    Args:
        embed_dim: Dimension of the embedding vectors
        vocab_size: Size of the vocabulary
        k: Number of top similarities to analyze
        num_trials: Number of simulation trials
    
    Returns:
        top_k_sims: Matrix of shape (num_trials, k) containing top-k similarities
                   for each trial
    """
    # Ensure k isn't larger than vocab_size
    if k > vocab_size:
        print(f"Warning: k ({k}) is larger than vocab_size ({vocab_size}). Setting k = vocab_size.")
        k = vocab_size
    
    top_k_sims = np.zeros((num_trials, k))
    
    for trial in tqdm(range(num_trials), desc="Simulation trials"):
        # Generate random query vector
        query = np.random.normal(0, 1, embed_dim)
        query = query / np.linalg.norm(query)
        
        # Generate random vocabulary vectors
        vocab = np.random.normal(0, 1, (vocab_size, embed_dim))
        # Normalize each vector to unit length
        norms = np.linalg.norm(vocab, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        vocab = vocab / norms
        
        # Compute similarities (cosine similarity)
        similarities = np.dot(vocab, query)
        
        # Get top-k similarities
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_sims[trial] = similarities[top_k_indices]
    
    return top_k_sims

def compute_real_top_k_similarities(model, tokenizer, query_words, k):
    """
    Compute top-k similarities between real query words and the model's vocabulary
    
    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        query_words: List of query words
        k: Number of top similarities to analyze
        
    Returns:
        top_k_sims: Matrix of shape (len(query_words), k) containing top-k similarities
                   for each query word
    """
    # A more robust approach for getting vocabulary embeddings
    print("Calculating vocabulary embeddings...")
    
    # Get the vocabulary
    vocab = list(tokenizer.get_vocab().keys())
    print(f"Vocabulary size: {len(vocab)}")
    
    # Process in smaller batches to avoid memory issues
    vocab_embeddings = []
    batch_size = 100
    
    for i in tqdm(range(0, len(vocab), batch_size)):
        batch_tokens = vocab[i:i+batch_size]
        # Filter out empty tokens and special tokens
        batch_tokens = [token for token in batch_tokens if token and len(token.strip()) > 0]
        
        if not batch_tokens:
            continue
            
        batch_embeddings = get_embeddings(model, tokenizer, batch_tokens)
        if batch_embeddings is not None:
            vocab_embeddings.append(batch_embeddings)
    
    if not vocab_embeddings:
        raise ValueError("Failed to compute any valid vocabulary embeddings.")
        
    vocab_embeddings = torch.cat(vocab_embeddings, dim=0)
    print(f"Processed {vocab_embeddings.shape[0]} vocabulary embeddings")
    
    # Compute similarities for each query word
    top_k_sims = np.zeros((len(query_words), k))
    
    for i, query in enumerate(tqdm(query_words, desc="Processing query words")):
        query_embedding = get_embeddings(model, tokenizer, query)
        if query_embedding is None:
            print(f"Warning: Could not get embedding for query '{query}'. Skipping.")
            top_k_sims[i] = np.zeros(k)
            continue
            
        # Make sure we get a valid shape for matrix multiplication
        if query_embedding.dim() == 2:
            similarities = torch.matmul(vocab_embeddings, query_embedding.T).squeeze()
        else:
            similarities = torch.matmul(vocab_embeddings, query_embedding)
        
        # Ensure k is not larger than the available vocabulary
        actual_k = min(k, similarities.shape[0])
        if actual_k < k:
            print(f"Warning: Only {actual_k} vocabulary embeddings available, requested top-{k}")
        
        # Get top-k similarities
        top_k_values, top_k_indices = torch.topk(similarities, actual_k)
        top_k_sims[i, :actual_k] = top_k_values.cpu().numpy()
        
        # Print top matches for this query
        print(f"\nTop {actual_k} matches for '{query}':")
        for j, idx in enumerate(top_k_indices):
            token_idx = idx.item()
            if token_idx < len(vocab):
                word = vocab[token_idx]
                print(f"{j+1}. {word}: {top_k_values[j]:.4f}")
            else:
                print(f"{j+1}. [Unknown token {token_idx}]: {top_k_values[j]:.4f}")
    
    return top_k_sims

def plot_top_k_distributions(top_k_sims, k, output_dir, is_real_data=False):
    """
    Plot histograms of top-k similarities with fitted Gaussian curves
    
    Args:
        top_k_sims: Matrix of top-k similarities
        k: Number of top similarities analyzed
        output_dir: Directory to save plots
        is_real_data: Whether the data is from real model or simulation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histograms of top-k similarities
    fig, axes = plt.subplots(k, 1, figsize=(10, 3*k))
    if k == 1:
        axes = [axes]
    
    for i in range(k):
        ax = axes[i]
        
        # Get data specific to this rank
        data = top_k_sims[:, i]
        
        # Create histogram
        n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, 
                                  color='skyblue', label=f'Empirical')
        
        # Fit Gaussian
        mu, std = norm.fit(data)
        
        # Only plot Gaussian fit if there are enough data points
        if len(data) > 5:
            x = np.linspace(data.min(), data.max(), 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'r-', linewidth=2, label=f'Gaussian fit: μ={mu:.4f}, σ={std:.4f}')
        
        ax.set_title(f'Top-{i+1} Similarity Distribution')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    source_type = "real_model" if is_real_data else "simulation"
    plt.savefig(f"{output_dir}/top_k_similarities_{source_type}.png", dpi=300)
    plt.close()
    
    # Create a single plot with all distributions
    plt.figure(figsize=(12, 8))
    for i in range(k):
        data = top_k_sims[:, i]
        mu, std = norm.fit(data)
        label = f'Top-{i+1}: μ={mu:.4f}, σ={std:.4f}'
        plt.hist(data, bins=30, density=True, alpha=0.3, label=label)
    
    plt.title(f'Comparison of Top-{k} Similarity Distributions')
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/top_k_similarities_comparison_{source_type}.png", dpi=300)
    plt.close()

def plot_similarity_decay(top_k_sims, output_dir, is_real_data=False):
    """Plot how similarity decreases as rank increases"""
    means = np.mean(top_k_sims, axis=0)
    stds = np.std(top_k_sims, axis=0)
    ranks = np.arange(1, len(means) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(ranks, means, yerr=stds, fmt='o-', capsize=5, elinewidth=1, markeredgewidth=1)
    plt.fill_between(ranks, means - stds, means + stds, alpha=0.2)
    
    plt.title('Similarity Decay with Rank')
    plt.xlabel('Rank')
    plt.ylabel('Average Similarity')
    plt.grid(alpha=0.3)
    
    # Fit exponential decay model
    from scipy.optimize import curve_fit
    
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    try:
        popt, _ = curve_fit(exp_decay, ranks, means)
        x_fit = np.linspace(1, len(means), 100)
        y_fit = exp_decay(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r-', label=f'Exp fit: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}')
        plt.legend()
    except:
        print("Could not fit exponential decay model.")
    
    source_type = "real_model" if is_real_data else "simulation"
    plt.savefig(f"{output_dir}/similarity_decay_{source_type}.png", dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    if args.use_real_model:
        print(f"Loading model {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=HUGGINGFACE_TOKEN)
        model = AutoModel.from_pretrained(args.model, token=HUGGINGFACE_TOKEN)
        
        # Fix padding token issue
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Set pad_token to eos_token")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print("Added [PAD] token as pad_token")
        
        # Compute similarities for real queries
        top_k_sims = compute_real_top_k_similarities(model, tokenizer, args.query_words, args.k)
    else:
        print("Running simulation...")
        # Run simulation
        top_k_sims = simulate_top_k_similarities(args.embed_dim, args.vocab_size, args.k, args.num_trials)
    
    # Plot results
    plot_top_k_distributions(top_k_sims, args.k, args.output_dir, args.use_real_model)
    plot_similarity_decay(top_k_sims, args.output_dir, args.use_real_model)
    
    # Calculate and print statistics
    means = np.mean(top_k_sims, axis=0)
    stds = np.std(top_k_sims, axis=0)
    
    print("\nSummary Statistics:")
    print("Rank | Mean Similarity | Std Dev")
    print("-" * 35)
    for i in range(args.k):
        print(f"{i+1:4d} | {means[i]:14.4f} | {stds[i]:7.4f}")
    
    # Save top-k similarity distributions to CSV
    np.savetxt(f"{args.output_dir}/top_k_similarities.csv", top_k_sims, delimiter=",")
    
    print(f"\nResults saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
    
### How to run 

## Mixtral 
# python nearest_word_stats.py --use_real_model --model "mistralai/Mistral-7B-v0.3" --query_words "computer" "ocean" "democracy" --k 10