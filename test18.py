import openai

# Set your OpenAI API key
openai.api_key = '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'

# Define two sentences for which you want to calculate cosine similarity
sentence1 = "The quick brown fox jumps over the lazy dog."
sentence2 = "A lazy dog is jumped over by the quick brown fox."

# Use OpenAI API to generate embeddings
response1 = openai.Completion.create(
    engine="text-embedding-ada-002",
    prompt=sentence1,
    max_tokens=0,
    n=1,
    stop=None,
    temperature=0,
)
embedding1 = response1['choices'][0]['text']

response2 = openai.Completion.create(
    engine="text-embedding-ada-002",
    prompt=sentence2,
    max_tokens=0,
    n=1,
    stop=None,
    temperature=0,
)
embedding2 = response2['choices'][0]['text']

# Calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    # Assuming embeddings are vectors
    # You may need to process the embeddings accordingly
    dot_product = sum(a*b for a, b in zip(embedding1, embedding2))
    magnitude1 = sum(a**2 for a in embedding1)**0.5
    magnitude2 = sum(b**2 for b in embedding2)**0.5
    return dot_product / (magnitude1 * magnitude2)

# Example usage
similarity_score = cosine_similarity(embedding1, embedding2)
print("Cosine Similarity Score:", similarity_score)
