import os
import json
import time
import telebot
import numpy as np
from threading import Thread
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import anthropic

load_dotenv()

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
ARTICLE_DIR = "cleaned_articles"
EMBED_FILE = "embeddings.json"
TOP_K = 5

# Initialize models
model = SentenceTransformer("all-MiniLM-L6-v2")
claude = anthropic.Anthropic(api_key=CLAUDE_API_KEY)


def create_embeddings():
    """Create embeddings for articles if they don't exist"""
    if os.path.exists(EMBED_FILE):
        return json.load(open(EMBED_FILE, encoding="utf-8"))

    print("Creating embeddings...")
    articles = []
    for file in os.listdir(ARTICLE_DIR):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(ARTICLE_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        embedding = model.encode(content).tolist()
        articles.append({
            "title": file.replace(".txt", ""),
            "content": content,
            "embedding": embedding
        })

    with open(EMBED_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f)

    return articles


def build_prompt(user_query, articles):
    """Find relevant articles and build prompt"""
    query_embedding = model.encode(user_query).reshape(1, -1)
    scores = []

    for article in articles:
        similarity = cosine_similarity(
            query_embedding,
            np.array(article["embedding"]).reshape(1, -1)
        )[0][0]
        scores.append((similarity, article))

    top_articles = sorted(scores, key=lambda x: x[0], reverse=True)[:TOP_K]
    context = "\n\n".join([f"{i + 1}. {a['content'].strip()}"
                           for i, (_, a) in enumerate(top_articles)])

    prompt = f"""You are a helpful assistant for Foodhub internal guides. Answer the user's question using the provided documentation.

- If the question is vague or lacks context, ask for clarification
- Be concise and easy to read on mobile
- List steps clearly if applicable
- Don't mention documents or sources

Question: {user_query.strip()}

Documentation:
{context}

Response:"""

    return prompt


def stream_response(chat_id, message_id, user_query, articles):
    """Generate and stream Claude response"""
    prompt = build_prompt(user_query, articles)
    response_text = ""
    last_update = time.time()

    try:
        with claude.messages.stream(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                response_text += text

                # Update message every 1.5 seconds
                if time.time() - last_update >= 1.5:
                    try:
                        bot.edit_message_text(response_text[:4096], chat_id, message_id)
                        last_update = time.time()
                    except Exception:
                        pass

        # Final update
        bot.edit_message_text(response_text[:4096], chat_id, message_id)

    except Exception as e:
        bot.edit_message_text(f"Error: {str(e)}", chat_id, message_id)


# Load articles
articles = create_embeddings()

# Initialize bot
bot = telebot.TeleBot(TELEGRAM_TOKEN)


@bot.message_handler(func=lambda m: True)
def handle_message(message):
    chat_id = message.chat.id
    user_input = message.text

    # Send initial response
    sent_msg = bot.send_message(chat_id, "Thinking...")
    message_id = sent_msg.message_id

    # Start streaming in background
    Thread(target=stream_response, args=(chat_id, message_id, user_input, articles)).start()


if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling()