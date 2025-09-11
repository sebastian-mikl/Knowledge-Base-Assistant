import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

ARTICLE_DIR = "cleaned_articles"
EMBEDDING_FILE = "embeddings.json"
TOP_K = 3

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings():
    """Create embeddings for all articles"""
    data = []
    for filename in os.listdir(ARTICLE_DIR):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(ARTICLE_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()

        embedding = model.encode(text)
        data.append({
            "title": filename.replace(".txt", ""),
            "embedding": embedding.tolist(),
            "content": text
        })

    with open(EMBEDDING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print("Embeddings created")


def load_embeddings():
    """Load pre-computed embeddings"""
    with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def search_articles(query):
    """Find most relevant articles for a query"""
    query_embedding = model.encode(query).reshape(1, -1)
    articles = load_embeddings()

    scores = []
    for article in articles:
        similarity = cosine_similarity(
            query_embedding,
            np.array(article["embedding"]).reshape(1, -1)
        )[0][0]
        scores.append((similarity, article))

    top_matches = sorted(scores, key=lambda x: x[0], reverse=True)[:TOP_K]

    print(f"\nTop {TOP_K} results for: \"{query}\"\n")
    for score, article in top_matches:
        print(f"Title: {article['title']} (Score: {score:.3f})")
        print(f"Preview: {article['content'][:300]}...")
        print("-" * 50)


if __name__ == "__main__":
    # Create embeddings if they don't exist
    if not os.path.exists(EMBEDDING_FILE):
        print("Creating embeddings...")
        create_embeddings()

    # Interactive search
    while True:
        query = input("\nEnter your question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        search_articles(query)