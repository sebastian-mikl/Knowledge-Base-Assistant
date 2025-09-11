# Knowledge Base Assistant

RAG-powered assistant for internal documentation with semantic search and Telegram bot interface.

## Problem

The companies ecosystem is vast - staff need to message multiple people in order to find the information they need. This information is entirely available in well documented knowledge base

## Solution

A retrieval-augmented generation (RAG) system that combines semantic search with AI-powered responses to provide accurate, contextual answers from the knowledge base.

**Key Features:**
- Semantic search using sentence transformers
- Advanced chunking with neighbor expansion for better context
- Real-time responses via Telegram bot
- Automated web scraping and text preprocessing
- Claude API integration for natural language responses

## Architecture

1. **Data Collection** (`scraper.py`) - Automated scraping of knowledge base articles
2. **Text Processing** (`cleaner.py`) - Removes metadata and formats content  
3. **Embedding Creation** (`retrieval.py`) - Semantic embeddings for similarity search
4. **Advanced Retrieval** (`chunked_retrieval.py`) - Context-aware chunking with neighbor expansion
5. **Bot Interface** (`telegram_bot.py`) - Real-time question answering via Telegram

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
playwright install
```

2. **Set up environment:**
```bash
cp .env.example .env
# Add your API keys to .env
```

3. **Collect and process documents:**
```bash
python scraper.py      # Scrape articles (requires manual scrolling)
python cleaner.py      # Clean and format text
```

4. **Test retrieval:**
```bash
python retrieval.py              # Basic similarity search
python chunked_retrieval.py      # Advanced context expansion
```

5. **Run Telegram bot:**
```bash
python telegram_bot.py
```

## Technical Details

**Semantic Search:**
- Uses `all-MiniLM-L6-v2` for document embeddings
- Cosine similarity for relevance ranking
- Caches embeddings for fast retrieval

**Context Enhancement:**
- Splits documents into 300-word chunks with 50-word overlap
- Expands top matches to include neighboring chunks
- Preserves document structure and context flow

**Response Generation:**
- Claude 3.5 Sonnet for natural language responses
- Streaming responses for better user experience
- Context-aware prompting with follow-up handling

## Usage

Ask questions like:
- "How do I reset a customer password?"
- "What are the refund policies?"
- "Steps to process a chargeback"

The system finds relevant documentation and provides clear answers.

## Requirements

- Python 3.8+, Telegram Bot Token, Claude API Key
- Run `pip install -r requirements.txt && playwright install`
- Copy `.env.example` to `.env` and add your API keys

Built for real documentation search problems with production considerations for caching, error handling, and user experience.