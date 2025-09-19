# LangChain Knowledge Base Assistant

A document retrieval system built to showcase LangChain's advanced features and LangSmith monitoring capabilities. This project demonstrates how to build production-ready RAG systems with conversation memory, hybrid retrieval, and comprehensive observability.

## Why LangChain and LangSmith

This project highlights practical applications of LangChain's ecosystem. Instead of building retrieval and memory systems from scratch, it leverages LangChain's mature components for document processing, vector search, and conversation management. LangSmith integration provides detailed tracing and performance monitoring that would be difficult to implement independently.

The system shows how these tools work together in a real application. Staff can ask questions about internal documentation through a Telegram bot, and the system maintains conversation context while retrieving relevant information. All interactions are automatically traced in LangSmith for debugging and optimization.

## Core LangChain Features

The project uses several key LangChain components. Document processing relies on RecursiveCharacterTextSplitter for intelligent text chunking and DirectoryLoader for batch document ingestion. The retrieval system combines multiple approaches using EnsembleRetriever, which blends semantic similarity search with keyword matching through BM25Retriever.

Conversation management uses ConversationBufferWindowMemory to maintain context across questions. Each user gets isolated memory that remembers recent exchanges without growing indefinitely. The system switches between simple RetrievalQA chains for standalone questions and ConversationalRetrievalChain for multi-turn conversations.

All components integrate with FAISS for vector storage and ChatAnthropic for response generation. The modular design makes it easy to swap different retrievers, memory types, or language models without rewriting the entire system.

## LangSmith Monitoring

LangSmith integration provides comprehensive observability without additional development effort. Every chain execution gets automatically traced, showing retrieval results, token usage, and response times. This visibility helps identify performance bottlenecks and optimize retrieval quality.

The system tracks which documents influence each response, making it easy to debug incorrect answers or improve document relevance. User interaction patterns become visible through the dashboard, revealing common question types and system usage trends.

Error tracking and performance monitoring happen automatically. When retrieval fails or responses take too long, LangSmith captures the details needed for troubleshooting. This level of observability would require significant custom development without LangSmith.

## Setup and Usage

Install the required packages and configure your environment:

```bash
pip install -r requirements_langchain.txt
cp .env.example .env
```

Add your API keys to the .env file. The LANGSMITH_API_KEY is optional but enables the monitoring features that make this project particularly useful for learning and development.

Test the LangChain components:

```bash
python test_langchain.py
```

Run the Telegram bot:

```bash
python langchain_telegram_bot.py
```

The system will create embeddings on first run, then cache them for faster subsequent starts. All interactions appear in your LangSmith dashboard if configured.

## Project Structure

The codebase demonstrates clean separation of LangChain components. Document processing logic lives in langchain_retrieval.py, conversation management in langchain_chains.py, and the bot interface in langchain_telegram_bot.py. Configuration centralizes model initialization and settings in config.py.

This structure makes it easy to understand how different LangChain components work together. The test suite shows how to validate each component independently, while the integration demonstrates their combined capabilities.

## What This Shows

This project proves that LangChain significantly reduces the complexity of building sophisticated RAG systems. Features like conversation memory, hybrid retrieval, and streaming responses become straightforward to implement using existing components.

LangSmith integration demonstrates why observability matters in production AI systems. The automatic tracing and performance monitoring provide insights that are difficult to obtain with custom logging solutions.

The combination shows how modern AI applications can be built efficiently using established frameworks rather than implementing everything from scratch.