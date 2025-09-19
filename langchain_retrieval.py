import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from config import get_embeddings, get_llm, ARTICLE_DIR, VECTOR_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP


class LangChainRetrieval:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        self.vectorstore = None
        self.retriever = None
        self.documents = None

    def load_documents(self):
        """Load all documents from the article directory"""
        print("Loading documents...")
        loader = DirectoryLoader(
            ARTICLE_DIR,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} documents")
        return self.documents

    def create_chunks(self):
        """Split documents into chunks using LangChain"""
        if not self.documents:
            self.load_documents()

        print("Creating text chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        chunks = text_splitter.split_documents(self.documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, force_recreate=False):
        """Create or load FAISS vector store"""
        if os.path.exists(VECTOR_STORE_PATH) and not force_recreate:
            print("Loading existing vector store...")
            self.vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True  # Safe since we created this file
            )
        else:
            print("Creating new vector store...")
            chunks = self.create_chunks()
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            # Save for future use
            self.vectorstore.save_local(VECTOR_STORE_PATH)
            print("Vector store saved")
        return self.vectorstore

    def create_basic_retriever(self, k=5):
        """Create basic semantic retriever"""
        if not self.vectorstore:
            self.create_vectorstore()
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def create_multi_query_retriever(self, k=5):
        """Create multi-query retriever that generates multiple search queries"""
        base_retriever = self.create_basic_retriever(k)
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
            parser_key="lines"
        )

    def create_hybrid_retriever(self, k=5):
        """Create ensemble retriever combining semantic and keyword search"""
        if not self.documents:
            self.load_documents()

        # Create semantic retriever
        semantic_retriever = self.create_basic_retriever(k)

        # Create BM25 keyword retriever
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = k

        # Combine with ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # Favor semantic search slightly
        )

        return ensemble_retriever

    def setup_retriever(self, retriever_type="basic"):
        """Setup retriever based on type"""
        if retriever_type == "multi_query":
            self.retriever = self.create_multi_query_retriever()
        elif retriever_type == "hybrid":
            self.retriever = self.create_hybrid_retriever()
        else:
            self.retriever = self.create_basic_retriever()
        return self.retriever

    def search(self, query, k=5):
        """Search for relevant documents"""
        if not self.retriever:
            self.setup_retriever()

        try:
            results = self.retriever.get_relevant_documents(query)
            return results[:k]
        except Exception as e:
            print(f"Search error: {e}")
            # Fallback to basic retriever
            basic_retriever = self.create_basic_retriever(k)
            return basic_retriever.get_relevant_documents(query)


def test_retrieval():
    """Test the retrieval system"""
    retrieval = LangChainRetrieval()

    # Test different retriever types
    test_query = "How do I reset a customer password?"

    print("Testing Basic Retriever:")
    retrieval.setup_retriever("basic")
    results = retrieval.search(test_query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('source', 'Unknown')}")
        print(f"   Preview: {doc.page_content[:200]}...")
        print()

    print("\n" + "=" * 50)
    print("Testing Multi-Query Retriever:")
    try:
        retrieval.setup_retriever("multi_query")
        results = retrieval.search(test_query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.metadata.get('source', 'Unknown')}")
            print(f"   Preview: {doc.page_content[:200]}...")
            print()
    except Exception as e:
        print(f"Multi-query retriever failed: {e}")

    print("\n" + "=" * 50)
    print("Testing Hybrid Retriever:")
    try:
        retrieval.setup_retriever("hybrid")
        results = retrieval.search(test_query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.metadata.get('source', 'Unknown')}")
            print(f"   Preview: {doc.page_content[:200]}...")
            print()
    except Exception as e:
        print(f"Hybrid retriever failed: {e}")


if __name__ == "__main__":
    test_retrieval()