import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
ARTICLE_DIR = "cleaned_articles"
EMBED_FILE = "chunked_embeddings.json"
TOP_MATCHES = 6
NEIGHBOR_RANGE = 2
CHUNK_SIZE = 300  # words per chunk
OVERLAP = 50  # word overlap between chunks

model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, size, overlap):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunk = " ".join(words[start:start + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def create_chunked_embeddings():
    """Create embeddings for text chunks"""
    all_chunks = []

    for fname in os.listdir(ARTICLE_DIR):
        if not fname.endswith(".txt"):
            continue

        with open(os.path.join(ARTICLE_DIR, fname), encoding="utf-8") as f:
            full_text = f.read()

        chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP)

        for idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            all_chunks.append({
                "title": fname.replace(".txt", ""),
                "chunk_id": idx,
                "content": chunk,
                "embedding": embedding
            })

    with open(EMBED_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)

    return all_chunks


def find_relevant_chunks(query):
    """Find chunks and expand with neighbors"""
    # Load or create embeddings
    if not os.path.exists(EMBED_FILE):
        print("Creating chunked embeddings...")
        all_chunks = create_chunked_embeddings()
    else:
        all_chunks = json.load(open(EMBED_FILE, encoding="utf-8"))

    # Score all chunks
    query_emb = model.encode(query).reshape(1, -1)
    scores = []

    for chunk in all_chunks:
        similarity = cosine_similarity(
            query_emb,
            np.array(chunk["embedding"]).reshape(1, -1)
        )[0][0]
        scores.append((similarity, chunk))

    # Get top matches
    top_chunks = sorted(scores, key=lambda x: x[0], reverse=True)[:TOP_MATCHES]

    # Expand to include neighboring chunks
    to_include = set()
    for _, chunk in top_chunks:
        title, chunk_id = chunk["title"], chunk["chunk_id"]
        for offset in range(-NEIGHBOR_RANGE, NEIGHBOR_RANGE + 1):
            to_include.add((title, chunk_id + offset))

    # Collect selected chunks in order
    selected = []
    seen = set()
    for chunk in all_chunks:
        key = (chunk["title"], chunk["chunk_id"])
        if key in to_include and key not in seen:
            selected.append(chunk)
            seen.add(key)

    return selected


def generate_context_prompt(query):
    """Generate a prompt with relevant context"""
    chunks = find_relevant_chunks(query)

    context = "\n\n".join(f"{i + 1}. {c['content'].strip()}"
                          for i, c in enumerate(chunks))

    prompt = f"""You are a helpful assistant for Foodhub internal guides. Answer the user's question using the provided documentation. Be concise and speak naturally.

Question: {query.strip()}

Context:
{context}

Answer:"""

    return prompt


if __name__ == "__main__":
    query = input("Enter your question: ")
    prompt = generate_context_prompt(query)
    print("\n" + "=" * 50)
    print(prompt)