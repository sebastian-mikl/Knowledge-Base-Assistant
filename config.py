import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# Configuration constants
ARTICLE_DIR = "cleaned_articles"
VECTOR_STORE_PATH = "vectorstore"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5
MEMORY_WINDOW = 5

def get_embeddings():
    """Get HuggingFace embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder="./models",
        encode_kwargs={'normalize_embeddings': True}
    )

def get_llm():
    """Get Claude LLM instance"""
    return ChatAnthropic(
        api_key=os.getenv("CLAUDE_API_KEY"),
        temperature=0
    )
def get_telegram_token():
    """Get Telegram bot token"""
    return os.getenv("TELEGRAM_TOKEN")

def get_claude_api_key():
    """Get Claude API key"""
    return os.getenv("CLAUDE_API_KEY")

def get_langsmith_integration():
    """Get LangSmith integration if available"""
    if os.getenv("LANGSMITH_API_KEY"):
        try:
            from langsmith_integration import LangSmithIntegration
            return LangSmithIntegration()
        except ImportError:
            print("LangSmith not available - continuing without tracing")
            return None
    return None