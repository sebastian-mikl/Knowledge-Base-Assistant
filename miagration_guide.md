# LangChain Migration Guide

This guide explains how to migrate from the original RAG system to the new LangChain-powered version.

## Branch Setup

```bash
# Create and switch to new branch
git checkout -b feature/langchain-integration

# Install new requirements
pip install -r requirements_langchain.txt
```

## File Changes Overview

### New Files Created
- `config.py` - Centralized configuration
- `langchain_retrieval.py` - LangChain-powered retrieval system
- `langchain_chains.py` - QA chains with memory
- `langchain_telegram_bot.py` - Enhanced Telegram bot
- `test_langchain.py` - Comprehensive test suite
- `requirements_langchain.txt` - Updated dependencies

### Files Replaced
- `retrieval.py` → `langchain_retrieval.py`
- `chunked_retrieval.py` → `langchain_retrieval.py` 
- `telegram_bot.py` → `langchain_telegram_bot.py`

### Files Unchanged
- `scraper.py` - No changes needed
- `cleaner.py` - No changes needed
- `.env.example` - No changes needed

## Key Improvements

### 1. Better Document Processing
**Before:**
```python
def chunk_text(text, size, overlap):
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunk = " ".join(words[start:start + size])
        chunks.append(chunk)
```

**After (LangChain):**
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```

### 2. Advanced Retrieval Methods
**Before:** Simple cosine similarity
```python
similarity = cosine_similarity(query_emb, chunk_emb)[0][0]
```

**After:** Multiple retrieval strategies
```python
# Semantic + Keyword hybrid
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# Multi-query expansion
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever, llm=llm
)
```

### 3. Conversation Memory
**Before:** No conversation history
**After:** Per-user conversation memory
```python
memory = ConversationBufferWindowMemory(
    k=5, memory_key="chat_history", return_messages=True
)
```

### 4. Better Error Handling
**Before:** Basic try/catch
**After:** Graceful fallbacks and comprehensive error handling

## Migration Steps

### Step 1: Setup Environment
```bash
# Backup current system
cp telegram_bot.py telegram_bot_backup.py
cp retrieval.py retrieval_backup.py

# Install new dependencies
pip install -r requirements_langchain.txt
```

### Step 2: Test New System
```bash
# Run comprehensive tests
python test_langchain.py

# Test specific components
python test_langchain.py docs
python test_langchain.py retriever
python test_langchain.py qa
```

### Step 3: Migrate Bot
```bash
# Test simple version first
python langchain_telegram_bot.py --simple

# Then test full version with streaming
python langchain_telegram_bot.py
```

## Performance Comparison

| Feature | Original System | LangChain System |
|---------|----------------|------------------|
| Chunking | Word-based splitting | Intelligent text splitting |
| Retrieval | Cosine similarity only | Hybrid (semantic + keyword) |
| Memory | None | Per-user conversation history |
| Query Expansion | None | Multi-query generation |
| Error Handling | Basic | Comprehensive with fallbacks |
| Scalability | Manual optimization | Built-in optimizations |

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'langchain'**
   ```bash
   pip install -r requirements_langchain.txt
   ```

2. **FAISS installation issues**
   ```bash
   # For CPU version
   pip install faiss-cpu
   
   # For GPU version (if needed)
   pip install faiss-gpu
   ```

3. **Memory issues with large document sets**
   - Use `create_vectorstore(force_recreate=True)` to rebuild
   - Reduce `CHUNK_SIZE` in config.py
   - Use basic retriever instead of hybrid

4. **Slow retrieval**
   - Check if vector store is saved/loaded properly
   - Use basic retriever for faster responses
   - Reduce `TOP_K` value

### Environment Variables
Make sure your `.env` file has:
```
TELEGRAM_TOKEN=your_bot_token
CLAUDE_API_KEY=your_claude_key
```

## Testing Checklist

- [ ] Documents load correctly
- [ ] Text chunking works
- [ ] Vector store creates/loads
- [ ] Basic retrieval works
- [ ] Hybrid retrieval works (if no errors)
- [ ] QA chains respond correctly
- [ ] Memory persists across questions
- [ ] Telegram bot responds
- [ ] Streaming works (advanced bot)
- [ ] Error handling graceful

## Rollback Plan

If issues arise, rollback with:
```bash
# Switch back to main branch
git checkout main

# Or rename backup files
mv telegram_bot_backup.py telegram_bot.py
mv retrieval_backup.py retrieval.py

# Use original requirements
pip install -r requirements.txt
```

## Next Steps After Migration

1. **Monitor Performance**: Compare response quality and speed
2. **Tune Parameters**: Adjust chunk sizes, retrieval methods
3. **Add Features**: 
   - Document metadata filtering
   - Query routing
   - Response caching
   - Analytics and logging
4. **Scale Up**: Consider vector databases for larger datasets

## Support

If you encounter issues:
1. Run `python test_langchain.py` for diagnostics
2. Check error logs for specific issues
3. Try the simple bot version first
4. Verify all requirements are installed correctly