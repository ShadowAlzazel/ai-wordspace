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

def get_closest_words(vector, embedding_layer, tokenizer, top_n=10, epsilon=1e-8):
    """Find closest words to a given vector using cosine similarity"""
    # Check for NaN or Inf values
    if torch.isnan(vector).any() or torch.isinf(vector).any():
        print("Warning: Input vector contains NaN or Inf values. Fixing...")
        vector = torch.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize the input vector 
    vector_norm = torch.norm(vector, p=2)
    if vector_norm > 0:
        vector = vector / vector_norm
    else:
        print("Warning: Zero vector detected")
        return [("ERROR_ZERO_VECTOR", 0.0)] * top_n
    
    # Normalize embeddings (row-wise)
    norms = torch.norm(embedding_layer.weight, p=2, dim=1, keepdim=True)
    valid_indices = norms.squeeze() > 0  # Filter out zero vectors
    normalized_embeddings = torch.zeros_like(embedding_layer.weight)
    normalized_embeddings[valid_indices] = embedding_layer.weight[valid_indices] / norms[valid_indices]
    
    # Compute cosine similarity
    similarity = torch.matmul(vector.unsqueeze(0), normalized_embeddings.T).squeeze(0)
    
    # Replace NaN values with very low similarity score
    similarity = torch.nan_to_num(similarity, nan=-1.0)
    
    # Filter out special tokens and control tokens
    valid_tokens = []
    for i in range(len(similarity)):
        token = tokenizer.decode([i]).strip()
        if not (token.startswith('[control_') or token in ['<unk>', '<s>', '</s>', '<pad>']):
            valid_tokens.append(i)
    
    # Get the top N most similar words from valid tokens
    valid_similarity = similarity[valid_tokens]
    if len(valid_similarity) >= top_n:
        top_values, relative_indices = torch.topk(valid_similarity, top_n)
        top_indices = [valid_tokens[idx] for idx in relative_indices]
    else:
        print(f"Warning: Only {len(valid_similarity)} valid tokens found, less than requested top_n={top_n}")
        top_values, relative_indices = torch.topk(valid_similarity, len(valid_similarity))
        top_indices = [valid_tokens[idx] for idx in relative_indices]
        # Pad with dummy entries
        while len(top_indices) < top_n:
            top_indices.append(-1)
            top_values = torch.cat([top_values, torch.tensor([-1.0], device=top_values.device)])
    
    # Decode and format results
    top_words = []
    top_scores = []
    for idx, val in zip(top_indices, top_values):
        if idx == -1:
            word = "N/A"
            score = -1.0
        else:
            word = tokenizer.decode([idx]).strip()
            score = val.item()
        top_words.append(word)
        top_scores.append(score)
    
    return list(zip(top_words, top_scores))

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze embedding space for word combinations')
    parser.add_argument('--words', nargs='+', required=True, 
                        help='Words to combine (e.g., --words tea milk)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_NAME,
                        help=f'HuggingFace model name (default: {DEFAULT_MODEL_NAME})')
    parser.add_argument('--semantic-words', nargs='+', default=None,
                        help='Specific words to track in bootstrap trials')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of bootstrap trials (default: 100)')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise level for bootstrap trials (default: 0.01)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of top words to retrieve (default: 20)')
    return parser.parse_args()

def get_word_vector(word, tokenizer, embedding_layer):
    """Get embedding vector for a word, handling multi-token words"""
    # Add space before if it's not a special token to improve tokenization
    if not word.startswith('[') and not word.startswith('<'):
        word_to_tokenize = ' ' + word
    else:
        word_to_tokenize = word
        
    word_ids = tokenizer.encode(word_to_tokenize, add_special_tokens=False)
    
    if not word_ids:
        print(f"Warning: '{word}' not found in vocabulary")
        return None
    
    # For multi-token words, average the embeddings
    tokens = tokenizer.convert_ids_to_tokens(word_ids)
    if len(word_ids) > 1:
        print(f"Note: '{word}' is tokenized into {len(word_ids)} tokens: {tokens}")
        vectors = []
        for token_id in word_ids:
            # Check if token is valid (not a special token)
            token = tokenizer.decode([token_id]).strip()
            if token and not (token.startswith('[control_') or token in ['<unk>', '<s>', '</s>', '<pad>']):
                vectors.append(embedding_layer(torch.tensor([token_id])))
        
        if not vectors:
            print(f"Error: No valid tokens found for '{word}'")
            return None
            
        vector = torch.mean(torch.stack(vectors), dim=0)
    else:
        # Check if the single token is valid
        token = tokenizer.decode([word_ids[0]]).strip()
        if token.startswith('[control_') or token in ['<unk>', '<s>', '</s>', '<pad>']:
            print(f"Warning: '{word}' maps to special token {token}, which may cause issues")
            
        vector = embedding_layer(torch.tensor([word_ids[0]]))
        
    # Check for NaN values
    if torch.isnan(vector).any():
        print(f"Warning: Vector for '{word}' contains NaN values. Replacing with zeros.")
        vector = torch.nan_to_num(vector, nan=0.0)
        
    return vector

def analyze_word_combination(words, semantic_set, model_name, trials=100, noise_level=0.01, top_n=20):
    """Analyze a combination of words in the embedding space"""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and model
    try:
        # First try with auth token
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=HUGGINGFACE_TOKEN,
            padding_side="right"
        )
        model = AutoModel.from_pretrained(
            model_name,
            use_auth_token=HUGGINGFACE_TOKEN,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
    except Exception as e:
        print(f"Warning: Failed to load with auth token, trying without: {e}")
        # Fall back to loading without auth token for open models
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right"
        )
        model = AutoModel.from_pretrained(
            model_name,
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
    
    # Get embeddings for each word
    word_vectors = []
    for word in words:
        vector = get_word_vector(word, tokenizer, embedding_layer)
        if vector is None:
            print(f"Skipping '{word}' due to tokenization issues.")
            continue
        word_vectors.append(vector)
    
    if not word_vectors:
        print("Error: No valid word vectors found. Cannot proceed.")
        return
        
    # Create composite vector
    composite_vector = torch.stack(word_vectors).sum(dim=0)
    
    # Check for NaN or Inf values in composite vector
    if torch.isnan(composite_vector).any() or torch.isinf(composite_vector).any():
        print("Warning: Composite vector contains NaN or Inf values. Fixing...")
        composite_vector = torch.nan_to_num(composite_vector, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Get initial closest words without noise
    print("\nClosest words to the combined vector:")
    closest_words = get_closest_words(composite_vector.squeeze(0), embedding_layer, tokenizer, top_n=top_n)
    for i, (word, sim) in enumerate(closest_words):
        print(f"{i+1:2d}. {word:<15} {sim:.4f}")
    
    # Store results from bootstrap trials
    T = trials
    semantic_ranks = {word: [] for word in semantic_set}
    semantic_sims = {word: [] for word in semantic_set}
    
    # Run bootstrap trials
    print(f"\nRunning {T} bootstrap trials with noise level {noise_level}...")
    for t in tqdm(range(T)):
        # Add small Gaussian noise
        noise = torch.randn_like(composite_vector) * noise_level
        noisy_vector = composite_vector + noise
        
        # Get closest words
        closest = get_closest_words(noisy_vector.squeeze(0), embedding_layer, tokenizer, top_n=top_n)
        
        # Create a dictionary mapping words to their ranks and similarities
        words_dict = {}
        for rank, (result_word, sim) in enumerate(closest):
            words_dict[result_word] = (rank, sim)
        
        # Check each word in semantic set
        for word in semantic_set:
            found = False
            for result_word, (rank, sim) in words_dict.items():
                if word.lower() in result_word.lower():  # Case-insensitive partial match
                    semantic_ranks[word].append(rank + 1)  # 1-indexed rank
                    semantic_sims[word].append(sim)
                    found = True
                    break
            if not found:
                # If word wasn't found, assign a rank beyond top_n and similarity of 0
                semantic_ranks[word].append(top_n + 1)
                semantic_sims[word].append(0)
    
    # Filter semantic words that were found in at least one trial
    found_words = [word for word in semantic_set if any(sim > 0 for sim in semantic_sims[word])]
    
    if len(found_words) >= 2:
        # Plot similarity distributions for found words
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray']
        
        for i, word in enumerate(found_words[:min(len(found_words), len(colors))]):
            sims = [s for s in semantic_sims[word] if s > 0]  # Filter out zeros
            if sims:
                mean_sim = np.mean(sims)
                std_sim = np.std(sims) if len(sims) > 1 else 0.01
                
                plt.hist(sims, bins=15, alpha=0.4, color=colors[i], 
                        label=f"{word}: μ={mean_sim:.2f}, σ={std_sim:.2f}")
                
                # Overlay fitted Gaussian
                if len(sims) > 5:  # Only show distribution if we have enough samples
                    x = np.linspace(max(0, min(sims)-0.05), max(sims)+0.05, 100)
                    plt.plot(x, norm.pdf(x, mean_sim, std_sim) * len(sims) * (max(sims)-min(sims))/15,
                            color=colors[i], linewidth=2)
        
        # Background distribution for reference
        x_bg = np.linspace(-0.1, 0.3, 100)
        bg_mean, bg_std = 0, 1/np.sqrt(embedding_dim)
        plt.plot(x_bg, norm.pdf(x_bg, bg_mean, bg_std) * 10,
                'k--', label=f"Background: N(0, 1/√D)")
        
        plt.title(f"'{' + '.join(words)}' Similarity Distribution over {T} Bootstrap Trials")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Print statistical summary
    print("\nStatistical Summary:")
    print(f"{'Word':<15} {'Mean Rank':<12} {'Rank Var':<12} {'Mean Sim':<12} {'Sim Var':<12} {'Found %':<10}")
    print("-" * 70)
    
    for word in semantic_set:
        ranks = semantic_ranks[word]
        sims = [s for s in semantic_sims[word] if s > 0]  # Only consider non-zero similarities
        found_percent = len(sims) / len(semantic_sims[word]) * 100
        
        if sims:  # Only report stats if word was found in at least one trial
            mean_rank = np.mean(ranks)
            var_rank = np.var(ranks)
            mean_sim = np.mean(sims)
            var_sim = np.var(sims) if len(sims) > 1 else 0
            print(f"{word:<15} {mean_rank:<12.2f} {var_rank:<12.2f} {mean_sim:<12.4f} {var_sim:<12.6f} {found_percent:<10.1f}")
        else:
            print(f"{word:<15} {'Not found':<12} {'-':<12} {'-':<12} {'-':<12} {0:<10.1f}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set default semantic words if not provided
    if args.semantic_words is None:
        # Add the input words and some related words as the default semantic set
        semantic_set = set(args.words)
        # Add a few common words that might be related (this is just an example)
        for word in args.words:
            semantic_set.add(word + "s")  # Add plural form as a simple heuristic
        semantic_set = list(semantic_set)
    else:
        semantic_set = args.semantic_words
    
    # Analyze the word combination
    analyze_word_combination(
        words=args.words,
        semantic_set=semantic_set,
        model_name=args.model,
        trials=args.trials,
        noise_level=args.noise,
        top_n=args.top_n
    )

if __name__ == "__main__":
    main()
    
### HOW TO RUN ###

## Basic example with tea and milk
# python closest_word_analyzer.py --words tea milk

## Complex example for Boba
# python closest_word_analyzer.py --words tea milk --semantic-words chai coffee latte

## Complex example with three words
# python closest_word_analyzer.py --words Asian Spaghetti Soup --semantic-words noodles broth ramen pho

## Political example with specific semantic tracking
# python closest_word_analyzer.py --words Germany Military --semantic-words Wehrmacht Bundeswehr army defense NATO tank panzer