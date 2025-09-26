# test_claude.py
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

# Test different parameter combinations
try:

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        anthropic_api_key=os.getenv("CLAUDE_API_KEY")
    )
    print("✓ anthropic_api_key + model works")
except Exception as e:
    print(f"✗ anthropic_api_key + model failed: {e}")

try:
    llm = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        anthropic_api_key=os.getenv("CLAUDE_API_KEY")
    )
    print("✓ anthropic_api_key + model_name works")
except Exception as e:
    print(f"✗ anthropic_api_key + model_name failed: {e}")

try:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("CLAUDE_API_KEY")
    )
    print("✓ api_key + model works")
except Exception as e:
    print(f"✗ api_key + model failed: {e}")