from transformers import AutoTokenizer, AutoModel
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
print(f"Hugging Face cache directory: {os.environ.get('HF_HOME') or '~/.cache/huggingface'}")


# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

def get_closest_word(vector, embedding_layer, tokenizer, top_n=1, epsilon=1e-8):
    # Normalize the input vector and embeddings for cosine similarity
    vector = F.normalize(vector.unsqueeze(0), p=2, dim=1, eps=epsilon)
    embeddings = F.normalize(embedding_layer.weight, p=2, dim=1, eps=epsilon)

    # Compute cosine similarity
    similarity = torch.matmul(vector, embeddings.T).squeeze(0)  # (1, vocab_size)

    # Get the top N most similar words
    top_indices = torch.topk(similarity, top_n).indices
    top_words = [tokenizer.decode([idx.item()]).strip() for idx in top_indices]
    top_scores = [similarity[idx].item() for idx in top_indices]

    return list(zip(top_words, top_scores))

# 1) Load tokenizer and embedding
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

def main() -> None:
    # Login
    token = HUGGINGFACE_TOKEN
    login(token=token)
    model_name = MODEL_NAME

    # Load model and tokenizer
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
    embedding_layer = model.get_input_embeddings()

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Access the embedding layer
    vocab_size, embedding_dim = embedding_layer.weight.shape
    print(f"Embedding layer has {vocab_size} tokens, each with {embedding_dim} dimensions.")

        # Example: Define your vector (e.g., sum of two token embeddings)
    word_1 = 'germany'
    word_2 = 'military'
    word1_id = tokenizer.convert_tokens_to_ids(word_1)
    word2_id = tokenizer.convert_tokens_to_ids(word_2)

    if word1_id is not None and word2_id is not None:
        word1_vector = embedding_layer(torch.tensor([word1_id]))
        word2_vector = embedding_layer(torch.tensor([word2_id]))

        # Check for zero vectors
        if torch.all(word1_vector == 0):
            print(f"Warning: Embedding vector for {word_1} is all zeros.")
        if torch.all(word2_vector == 0):
            print(f"Warning: Embedding vector for {word_2} is all zeros.")

        combined_vector = word1_vector + word2_vector

        # Print the resulting vector
        print("Combined Vector:", combined_vector)

        # Find closest words
        top_n = 10
        closest_words_with_scores = get_closest_word(combined_vector.squeeze(0), embedding_layer, tokenizer, top_n=top_n)
        print("Closest words:", closest_words_with_scores)

        if closest_words_with_scores and not any(np.isnan(score) for _, score in closest_words_with_scores):
            words = [item[0] for item in closest_words_with_scores]
            scores = [item[1] for item in closest_words_with_scores]

            # Convert cosine similarity scores to a distance metric (1 - cosine similarity)
            distances = [1 - score for score in scores]
            mean_distance = np.mean(distances)

            # Create the binomial distribution data (for visualization purposes)
            probabilities = [1 / (1 + dist) for dist in distances]
            n_trials = 100  # Number of trials for the binomial distribution simulation
            binomial_samples = []
            for p in probabilities:
                samples = np.random.binomial(1, p, n_trials)
                binomial_samples.extend(samples)

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.hist(binomial_samples, bins=2, edgecolor='black', alpha=0.7)
            plt.xticks([0, 1], ['Other Words', 'Nearby Words'])
            plt.title(f'Distribution of Nearby Words (Binomial Simulation)\nMean Distance: {mean_distance:.4f}')
            plt.xlabel('Word Category')
            plt.ylabel('Frequency')
            plt.grid(axis='y', alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print("No valid closest words found (likely due to NaN scores).")
    else:
        print("One or both of the input words ('tea', 'milk') are not in the tokenizer's vocabulary.")

if __name__ == "__main__":
    main()