from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from config import get_llm, MEMORY_WINDOW, TOP_K, get_langsmith_integration


class LangChainQA:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = get_llm()
        self.user_memories = {}
        self.langsmith = get_langsmith_integration()  # Add LangSmith integration

    def create_qa_chain(self):
        """Create a simple QA chain without memory"""
        template = """You are a helpful assistant for Foodhub internal guides. Answer the user's question using the provided documentation. Be concise and speak naturally.

- If the question is vague or lacks context, ask for clarification
- Be concise and easy to read on mobile
- List steps clearly if applicable
- Don't mention documents or sources unless specifically asked

Use the following context to answer the question:

{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )

        return qa_chain

    def get_user_memory(self, user_id):
        """Get or create memory for a specific user"""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationBufferWindowMemory(
                k=MEMORY_WINDOW,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        return self.user_memories[user_id]

    def create_conversational_chain(self, user_id):
        """Create conversational QA chain with memory for specific user"""
        memory = self.get_user_memory(user_id)

        template = """You are a helpful assistant for Foodhub internal guides. Answer the user's question using the provided documentation and chat history.

- If the question is vague or lacks context, ask for clarification
- Be concise and easy to read on mobile
- List steps clearly if applicable
- Don't mention documents or sources unless specifically asked
- Use the chat history to provide contextual responses

Chat History:
{chat_history}

Context from documentation:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False
        )

        return conversational_chain

    def ask_question(self, question, user_id=None, use_memory=True):
        """Ask a question and get an answer"""
        try:
            # Get LangSmith config if available
            config = None
            if self.langsmith and user_id:
                config = self.langsmith.get_runnable_config(user_id)

            if use_memory and user_id:
                chain = self.create_conversational_chain(user_id)
                if config:
                    response = chain.invoke({"question": question}, config=config)
                else:
                    response = chain({"question": question})

                # Log retrieval results if LangSmith is available
                if self.langsmith and "source_documents" in response:
                    self.langsmith.log_retrieval_results(
                        question, response.get("source_documents", [])
                    )

                return response["answer"]
            else:
                chain = self.create_qa_chain()
                if config:
                    response = chain.invoke({"query": question}, config=config)
                    result = response["result"]
                else:
                    response = chain({"query": question})
                    result = response["result"]

                # Log retrieval results if LangSmith is available
                if self.langsmith and "source_documents" in response:
                    self.langsmith.log_retrieval_results(
                        question, response.get("source_documents", [])
                    )

                return result

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def clear_user_memory(self, user_id):
        """Clear memory for a specific user"""
        if user_id in self.user_memories:
            self.user_memories[user_id].clear()

    def get_user_conversation_history(self, user_id):
        """Get conversation history for a user"""
        if user_id in self.user_memories:
            memory = self.user_memories[user_id]
            return memory.chat_memory.messages
        return []


class StreamingQA:
    """For streaming responses (used with Telegram bot)"""

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = get_llm()

    def build_prompt(self, question, user_id=None, chat_history=None):
        """Build prompt with retrieved context"""
        # Get relevant documents
        try:
            docs = self.retriever.get_relevant_documents(question)
            context = "\n\n".join([f"{i + 1}. {doc.page_content.strip()}"
                                   for i, doc in enumerate(docs[:TOP_K])])
        except Exception as e:
            print(f"Retrieval error: {e}")
            context = "No relevant documentation found."

        # Build prompt with or without history
        if chat_history:
            history_text = "\n".join([
                f"{'Human' if isinstance(msg, BaseMessage) and msg.type == 'human' else 'Assistant'}: {msg.content}"
                for msg in chat_history[-6:]  # Last 6 messages
            ])

            prompt = f"""You are a helpful assistant for Foodhub internal guides. Answer the user's question using the provided documentation and chat history.

- If the question is vague or lacks context, ask for clarification
- Be concise and easy to read on mobile
- List steps clearly if applicable
- Don't mention documents or sources unless specifically asked
- Use the chat history to provide contextual responses

Previous conversation:
{history_text}

Documentation:
{context}

Question: {question}

Answer:"""
        else:
            prompt = f"""You are a helpful assistant for Foodhub internal guides. Answer the user's question using the provided documentation.

- If the question is vague or lacks context, ask for clarification
- Be concise and easy to read on mobile
- List steps clearly if applicable
- Don't mention documents or sources unless specifically asked

Documentation:
{context}

Question: {question}

Answer:"""

        return prompt


def test_chains():
    """Test the QA chains"""
    from langchain_retrieval import LangChainRetrieval

    # Setup retrieval
    retrieval = LangChainRetrieval()
    retrieval.setup_retriever("basic")

    # Setup QA
    qa = LangChainQA(retrieval.retriever)

    # Test questions
    questions = [
        "How do I reset a customer password?",
        "What are the refund policies?",
        "Can you help me with payment processing?"
    ]

    print("Testing Simple QA Chain:")
    for q in questions:
        answer = qa.ask_question(q, use_memory=False)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

    print("\n" + "=" * 50)
    print("Testing Conversational Chain:")
    user_id = "test_user"

    # Simulate conversation
    conversation = [
        "How do I process a refund?",
        "What if the customer paid with a credit card?",
        "How long does it take?"
    ]

    for q in conversation:
        answer = qa.ask_question(q, user_id=user_id, use_memory=True)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    test_chains()