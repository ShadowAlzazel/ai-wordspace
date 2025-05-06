import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from scipy.stats import binom, chi2
from dotenv import load_dotenv

# Suppress HF logs for brevity
from transformers import logging
logging.set_verbosity_error()

# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

def load_embeddings(model_name, token=None):
    """Load tokenizer and model embeddings"""
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

def run_threshold_experiment(ref_token="tea", tau=0.5, n=100, T=200):
    """Run the thresholding experiment as described in Section 3.2"""
    print(f"Loading tokenizer and embedding layer for {MODEL_NAME}...")
    tokenizer, embedding_layer = load_embeddings(MODEL_NAME, HUGGINGFACE_TOKEN)
    vocab_size, D = embedding_layer.weight.shape
    
    # Get reference embedding for the token
    ref_id = tokenizer.convert_tokens_to_ids(ref_token)
    ref_vec = F.normalize(embedding_layer.weight[ref_id], dim=0)
    
    print(f"Running {T} trials with n={n} samples each, threshold τ={tau}...")
    counts = []
    all_sims = []
    
    # Perform T trials
    for t in range(T):
        if t % 20 == 0:
            print(f"Trial {t}/{T}...")
        
        # Sample n random tokens
        sample_ids = random.sample(range(vocab_size), n)
        
        # Compute cosine similarities in batch
        batch = F.normalize(embedding_layer.weight[sample_ids], dim=1)
        sims = (batch @ ref_vec).cpu().detach().numpy()
        
        # Count how many similarities exceed threshold
        trial_successes = (sims > tau).sum()
        counts.append(trial_successes)
        all_sims.extend(sims)
    
    # Calculate empirical probability
    p_hat = np.mean([c / n for c in counts])
    p_direct = np.mean(np.array(all_sims) > tau)
    
    print(f"Estimated p̂ = {p_hat:.4f} (from counts)")
    print(f"Estimated p̂ = {p_direct:.4f} (direct from all similarities)")
    
    # Calculate empirical PMF 
    values, freqs = np.unique(counts, return_counts=True)
    emp_pmf = freqs / T
    
    # Calculate theoretical PMF
    theo_pmf = binom.pmf(values, n, p_hat)
    
    # Calculate goodness-of-fit (chi-square statistic)
    valid_indices = theo_pmf > 0
    chi_square = np.sum(((emp_pmf[valid_indices] - theo_pmf[valid_indices])**2) / theo_pmf[valid_indices])
    dof = len(values[valid_indices]) - 1  # degrees of freedom
    p_value = 1 - chi2.cdf(chi_square, dof)
    
    print(f"Chi-square statistic: {chi_square:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    
    # Hypothesis test interpretation
    alpha = 0.05
    if p_value > alpha:
        print(f"At significance level {alpha}, the data appears consistent with a binomial distribution.")
    else:
        print(f"At significance level {alpha}, the data deviates significantly from a binomial distribution.")
    
    return values, emp_pmf, theo_pmf, p_hat, n, counts

def plot_results(values, emp_pmf, theo_pmf, p_hat, n, counts):
    """Plot the empirical and theoretical PMFs"""
    plt.figure(figsize=(10, 6))
    
    # Plot empirical PMF as bars
    plt.bar(values, emp_pmf, alpha=0.6, label='Empirical frequency')
    
    # Plot theoretical PMF as points and lines
    plt.plot(values, theo_pmf, 'ro-', label=f'Binomial(n={n}, p̂={p_hat:.3f})')
    
    plt.xlabel('Number of successes (k)')
    plt.ylabel('Probability P(X=k)')
    plt.title(f'Threshold Similarity Success Counts vs. Binomial PMF (n={n}, τ=0.5)')
    plt.legend()
    
    # Add a histogram of the raw counts as an inset
    ax2 = plt.axes([0.65, 0.25, 0.25, 0.25])
    ax2.hist(counts, bins=min(20, len(set(counts))), alpha=0.7)
    ax2.set_title('Raw counts histogram')
    
    plt.tight_layout()
    plt.savefig('similarity_binomial_results.png')
    plt.show()

def main():
    # Run the experiment with parameters as in the paper
    values, emp_pmf, theo_pmf, p_hat, n, counts = run_threshold_experiment(
        ref_token="tea",
        tau=0.5,
        n=100,
        T=200
    )
    
    # Plot the results
    plot_results(values, emp_pmf, theo_pmf, p_hat, n, counts)
    
    # Optional: Save the results
    result_dict = {
        "values": values.tolist(),
        "empirical_pmf": emp_pmf.tolist(),
        "theoretical_pmf": theo_pmf.tolist(),
        "p_hat": p_hat,
        "n": n,
        "counts": counts
    }
    
    # Save as numpy file for later analysis
    np.save("binomial_results.npy", result_dict)
    print("Results saved to binomial_results.npy")

if __name__ == "__main__":
    main()