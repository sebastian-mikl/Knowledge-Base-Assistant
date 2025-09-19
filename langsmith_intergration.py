import os
from langsmith import Client
from langsmith.wrappers import wrap_openai
from langchain.callbacks import LangChainTracer
from langchain.schema.runnable import RunnableConfig
from dotenv import load_dotenv

load_dotenv()


class LangSmithIntegration:
    def __init__(self):
        # Initialize LangSmith client
        self.client = Client(
            api_url="https://api.smith.langchain.com",
            api_key=os.getenv("LANGSMITH_API_KEY")
        )

        # Setup tracer
        self.tracer = LangChainTracer(
            project_name="Knowledge-Base-Assistant",
            client=self.client
        )

    def get_runnable_config(self, user_id=None, session_id=None):
        """Get configuration for tracing LangChain runs"""
        tags = ["rag", "telegram-bot", "foodhub"]

        if user_id:
            tags.append(f"user:{user_id}")

        metadata = {
            "environment": "development",  # or "production"
            "version": "2.0-langchain"
        }

        if session_id:
            metadata["session_id"] = session_id

        return RunnableConfig(
            callbacks=[self.tracer],
            tags=tags,
            metadata=metadata
        )

    def log_retrieval_results(self, query, retrieved_docs, scores=None):
        """Log retrieval results for analysis"""
        inputs = {"query": query}
        outputs = {
            "retrieved_count": len(retrieved_docs),
            "documents": [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata,
                    "score": scores[i] if scores else None
                }
                for i, doc in enumerate(retrieved_docs)
            ]
        }

        self.client.create_run(
            name="document_retrieval",
            project_name="Knowledge-Base-Assistant",
            run_type="retriever",
            inputs=inputs,
            outputs=outputs
        )

    def log_user_feedback(self, run_id, score, feedback_text=None):
        """Log user feedback for a specific response"""
        self.client.create_feedback(
            run_id=run_id,
            key="user_satisfaction",
            score=score,  # 0-1 scale
            comment=feedback_text
        )

    def create_dataset(self, name, description="Test cases for RAG system"):
        """Create a dataset for evaluation"""
        return self.client.create_dataset(
            dataset_name=name,
            description=description
        )

    def add_test_cases(self, dataset_name):
        """Add test cases to dataset"""
        test_cases = [
            {
                "inputs": {"question": "How do I reset a customer password?"},
                "outputs": {"answer": "Expected answer about password reset procedure"}
            },
            {
                "inputs": {"question": "What are the refund policies?"},
                "outputs": {"answer": "Expected answer about refund policies"}
            },
            {
                "inputs": {"question": "How do I process a chargeback?"},
                "outputs": {"answer": "Expected answer about chargeback process"}
            }
        ]

        dataset = self.client.read_dataset(dataset_name=dataset_name)

        for case in test_cases:
            self.client.create_example(
                inputs=case["inputs"],
                outputs=case["outputs"],
                dataset_id=dataset.id
            )


# Modified config.py to include LangSmith
def get_langsmith_config():
    """Get LangSmith configuration if enabled"""
    if os.getenv("LANGSMITH_API_KEY"):
        return LangSmithIntegration()
    return None


# Add to your existing chains
def create_traced_qa_chain(retriever, langsmith_integration=None):
    """Create QA chain with LangSmith tracing"""
    from langchain.chains import RetrievalQA
    from config import get_llm

    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        return_source_documents=True
    )

    # If LangSmith is available, return chain with config
    if langsmith_integration:
        def traced_qa(question, user_id=None):
            config = langsmith_integration.get_runnable_config(user_id)
            return qa_chain.invoke({"query": question}, config=config)

        return traced_qa

    # Otherwise return normal chain
    return lambda question, user_id=None: qa_chain.invoke({"query": question})


# Usage in telegram bot
def handle_message_with_tracing(message, qa_chain, langsmith_integration=None):
    """Handle message with optional LangSmith tracing"""
    user_id = str(message.from_user.id)
    query = message.text

    if langsmith_integration:
        # Use traced version
        config = langsmith_integration.get_runnable_config(user_id)
        response = qa_chain.invoke({"query": query}, config=config)

        # Log retrieval results
        if "source_documents" in response:
            langsmith_integration.log_retrieval_results(
                query, response["source_documents"]
            )
    else:
        # Use normal version
        response = qa_chain.invoke({"query": query})

    return response["result"]