#!/usr/bin/env python3
"""
Test script for LangChain RAG system
Run this to verify everything is working correctly
"""

import os
import sys
from langchain_retrieval import LangChainRetrieval
from langchain_chains import LangChainQA


def test_document_loading():
    """Test document loading functionality"""
    print("🔍 Testing Document Loading...")

    retrieval = LangChainRetrieval()
    try:
        documents = retrieval.load_documents()
        print(f"Loaded {len(documents)} documents")

        if documents:
            print(f"Sample document: {documents[0].metadata}")
            print(f"Content preview: {documents[0].page_content[:200]}...")
        return True
    except Exception as e:
        print(f"Document loading failed: {e}")
        return False


def test_chunking():
    """Test text chunking functionality"""
    print("\nTesting Text Chunking...")

    retrieval = LangChainRetrieval()
    try:
        chunks = retrieval.create_chunks()
        print(f"Created {len(chunks)} chunks")

        if chunks:
            print(f"Sample chunk metadata: {chunks[0].metadata}")
            print(f"Chunk content: {chunks[0].page_content[:150]}...")
        return True
    except Exception as e:
        print(f"Chunking failed: {e}")
        return False


def test_vector_store():
    """Test vector store creation and loading"""
    print("\nTesting Vector Store...")

    retrieval = LangChainRetrieval()
    try:
        vectorstore = retrieval.create_vectorstore()
        print("Vector store created/loaded successfully")

        # Test similarity search
        test_query = "password reset"
        results = vectorstore.similarity_search(test_query, k=3)
        print(f"🔍 Similarity search for '{test_query}' returned {len(results)} results")

        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.metadata.get('source', 'Unknown')[:50]}...")

        return True
    except Exception as e:
        print(f"Vector store test failed: {e}")
        return False


def test_retrievers():
    """Test different retriever types"""
    print("\nTesting Retrievers...")

    retrieval = LangChainRetrieval()
    test_query = "How do I process a refund?"

    retriever_types = ["basic", "multi_query", "hybrid"]
    results = {}

    for ret_type in retriever_types:
        try:
            print(f"   Testing {ret_type} retriever...")
            retrieval.setup_retriever(ret_type)
            docs = retrieval.search(test_query, k=3)
            results[ret_type] = len(docs)
            print(f"   {ret_type}: {len(docs)} documents retrieved")
        except Exception as e:
            print(f"   {ret_type} failed: {e}")
            results[ret_type] = 0

    return any(count > 0 for count in results.values())


def test_qa_chains():
    """Test QA chain functionality"""
    print("\nTesting QA Chains...")

    try:
        # Setup retrieval
        retrieval = LangChainRetrieval()
        retrieval.setup_retriever("basic")

        # Setup QA
        qa = LangChainQA(retrieval.retriever)

        # Test simple QA
        print("   Testing simple QA...")
        question = "How do I reset a password?"
        answer = qa.ask_question(question, use_memory=False)
        print(f"   Q: {question}")
        print(f"   A: {answer[:100]}...")

        # Test conversational QA
        print("   Testing conversational QA...")
        user_id = "test_user"

        questions = [
            "How do I process a refund?",
            "What if the payment was made with a credit card?",
            "How long does this usually take?"
        ]

        for q in questions:
            answer = qa.ask_question(q, user_id=user_id, use_memory=True)
            print(f"   Q: {q}")
            print(f"   A: {answer[:100]}...")

        print("QA chains working correctly")
        return True

    except Exception as e:
        print(f"QA chains test failed: {e}")
        return False


def test_memory():
    """Test memory functionality"""
    print("\nTesting Memory System...")

    try:
        retrieval = LangChainRetrieval()
        retrieval.setup_retriever("basic")
        qa = LangChainQA(retrieval.retriever)

        user_id = "memory_test_user"

        # Ask first question
        q1 = "What are the refund policies?"
        a1 = qa.ask_question(q1, user_id=user_id, use_memory=True)

        # Ask follow-up
        q2 = "Can you give me more details about that?"
        a2 = qa.ask_question(q2, user_id=user_id, use_memory=True)

        # Check memory
        history = qa.get_user_conversation_history(user_id)

        print(f"Memory test completed")
        print(f"   First question: {q1}")
        print(f"   Follow-up: {q2}")
        print(f"   History length: {len(history)} messages")

        # Clear memory
        qa.clear_user_memory(user_id)
        history_after_clear = qa.get_user_conversation_history(user_id)
        print(f"   History after clear: {len(history_after_clear)} messages")

        return True

    except Exception as e:
        print(f"Memory test failed: {e}")
        return False


def test_integration():
    """Test full integration"""
    print("\nTesting Full Integration...")

    try:
        # Test the complete pipeline
        retrieval = LangChainRetrieval()
        retrieval.setup_retriever("hybrid")
        qa = LangChainQA(retrieval.retriever)

        # Simulate real user interaction
        test_scenarios = [
            {
                "user": "new_user_1",
                "questions": [
                    "How do I handle customer complaints?",
                    "What should I do if they're not satisfied with my response?"
                ]
            },
            {
                "user": "new_user_2",
                "questions": [
                    "Steps to process a chargeback",
                    "How long does the chargeback process take?",
                    "What documentation do I need?"
                ]
            }
        ]

        for scenario in test_scenarios:
            user_id = scenario["user"]
            print(f"   Testing scenario for {user_id}:")

            for question in scenario["questions"]:
                answer = qa.ask_question(question, user_id=user_id, use_memory=True)
                print(f"     Q: {question}")
                print(f"     A: {answer[:80]}...")

        print("Integration test completed successfully")
        return True

    except Exception as e:
        print(f"Integration test failed: {e}")
        return False


def check_requirements():
    """Check if all required files and directories exist"""
    print("Checking Requirements...")

    required_dirs = ["cleaned_articles"]
    required_files = [".env"]

    missing_items = []

    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_items.append(f"Directory: {directory}")
        else:
            file_count = len([f for f in os.listdir(directory) if f.endswith('.txt')])
            print(f"Found {file_count} .txt files in {directory}/")

    for file in required_files:
        if not os.path.exists(file):
            missing_items.append(f"File: {file}")

    if missing_items:
        print("Missing required items:")
        for item in missing_items:
            print(f"   - {item}")
        return False

    print("All requirements satisfied")
    return True


def run_all_tests():
    """Run all tests in sequence"""
    print("LangChain RAG System Test Suite")
    print("=" * 50)

    if not check_requirements():
        print("\nRequirements check failed. Please ensure you have:")
        print("   - cleaned_articles/ directory with .txt files")
        print("   - .env file with API keys")
        return

    tests = [
        ("Document Loading", test_document_loading),
        ("Text Chunking", test_chunking),
        ("Vector Store", test_vector_store),
        ("Retrievers", test_retrievers),
        ("QA Chains", test_qa_chains),
        ("Memory System", test_memory),
        ("Full Integration", test_integration)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print("\nTests interrupted by user")
            break
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   📈 Success Rate: {passed / (passed + failed) * 100:.1f}%" if (passed + failed) > 0 else "")

    if failed == 0:
        print("\nAll tests passed! Your LangChain RAG system is ready.")
    else:
        print(f"\n{failed} test(s) failed. Check the error messages above.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1].lower()
        test_map = {
            "docs": test_document_loading,
            "chunks": test_chunking,
            "vector": test_vector_store,
            "retriever": test_retrievers,
            "qa": test_qa_chains,
            "memory": test_memory,
            "integration": test_integration
        }

        if test_name in test_map:
            if check_requirements():
                test_map[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {', '.join(test_map.keys())}")
    else:
        # Run all tests
        run_all_tests()