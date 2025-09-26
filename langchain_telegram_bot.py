import os
import time
import telebot
from threading import Thread
from langchain.schema import HumanMessage, AIMessage
import anthropic
from config import get_telegram_token, get_claude_api_key, get_langsmith_integration
from langchain_retrieval import LangChainRetrieval
from langchain_chains import LangChainQA, StreamingQA


class LangChainTelegramBot:
    def __init__(self):
        self.bot = telebot.TeleBot(get_telegram_token())
        self.claude = anthropic.Anthropic(api_key=get_claude_api_key())
        self.langsmith = get_langsmith_integration()  # Add LangSmith integration

        # Setup retrieval system
        print("Setting up LangChain retrieval system...")
        self.retrieval = LangChainRetrieval()
        self.retrieval.setup_retriever("hybrid")  # Use hybrid retriever

        # Setup QA system
        self.qa_system = LangChainQA(self.retrieval.retriever)
        self.streaming_qa = StreamingQA(self.retrieval.retriever)

        # Setup bot handlers
        self.setup_handlers()

        if self.langsmith:
            print("🔍 LangSmith tracing enabled!")
        else:
            print("ℹ️ LangSmith not configured - running without tracing")

        print("Bot initialized successfully!")

    def setup_handlers(self):
        """Setup Telegram bot message handlers"""

        @self.bot.message_handler(commands=['start'])
        def handle_start(message):
            welcome_msg = """🤖 Welcome to the Foodhub Knowledge Assistant!

I can help you find information from our internal documentation. Just ask me any question about:

• Account management
• Payment processing  
• Refund policies
• Technical procedures
• And much more!

Try asking: "How do I reset a customer password?"

💡 I remember our conversation, so you can ask follow-up questions!

Commands:
/clear - Clear our conversation history
/help - Show this message"""

            self.bot.send_message(message.chat.id, welcome_msg)

        @self.bot.message_handler(commands=['help'])
        def handle_help(message):
            help_msg = """🔍 How to use this bot:

1. Ask any question about Foodhub procedures
2. I'll search our knowledge base and provide answers
3. Ask follow-up questions - I remember our conversation!

Examples:
• "How do I process a refund?"
• "What's the policy for chargebacks?"
• "Steps to verify a customer account"

Commands:
/clear - Start fresh conversation
/start - Show welcome message"""

            self.bot.send_message(message.chat.id, help_msg)

        @self.bot.message_handler(commands=['clear'])
        def handle_clear(message):
            user_id = str(message.from_user.id)
            self.qa_system.clear_user_memory(user_id)
            self.bot.send_message(message.chat.id, "✅ Conversation history cleared! What would you like to know?")

        @self.bot.message_handler(func=lambda m: True)
        def handle_message(message):
            chat_id = message.chat.id
            user_id = str(message.from_user.id)
            user_input = message.text

            # Send initial "thinking" message
            sent_msg = self.bot.send_message(chat_id, "Thinking..")
            message_id = sent_msg.message_id

            # Start streaming response in background
            Thread(
                target=self.stream_response,
                args=(chat_id, message_id, user_input, user_id)
            ).start()

    def stream_response(self, chat_id, message_id, user_query, user_id):
        """Generate and stream Claude response using LangChain"""
        try:
            # Get conversation history from LangChain memory
            chat_history = self.qa_system.get_user_conversation_history(user_id)

            # Build prompt with LangChain retrieval
            prompt = self.streaming_qa.build_prompt(user_query, user_id, chat_history)

            # Stream response from Claude
            response_text = ""
            last_update = time.time()

            with self.claude.messages.stream(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    response_text += text

                    # Update message every 1.5 seconds
                    if time.time() - last_update >= 1.5:
                        try:
                            self.bot.edit_message_text(
                                response_text[:4096],
                                chat_id,
                                message_id
                            )
                            last_update = time.time()
                        except Exception:
                            pass

            # Final update
            final_response = response_text[:4096]
            self.bot.edit_message_text(final_response, chat_id, message_id)

            # Update LangChain memory manually
            memory = self.qa_system.get_user_memory(user_id)
            memory.chat_memory.add_user_message(user_query)
            memory.chat_memory.add_ai_message(final_response)

        except Exception as e:
            error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
            self.bot.edit_message_text(error_msg, chat_id, message_id)

    def run(self):
        """Start the bot"""
        print("🚀 LangChain Telegram bot is running...")
        print("Press Ctrl+C to stop")
        try:
            self.bot.infinity_polling(timeout=60, long_polling_timeout=60)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"❌ Bot error: {e}")


class SimpleLangChainBot:
    """Simpler version using LangChain chains directly (non-streaming)"""

    def __init__(self):
        self.bot = telebot.TeleBot(get_telegram_token())

        # Setup LangChain components
        print("Setting up LangChain components...")
        self.retrieval = LangChainRetrieval()
        self.retrieval.setup_retriever("basic")  # Basic retriever for simplicity
        self.qa_system = LangChainQA(self.retrieval.retriever)

        self.setup_handlers()
        print("Simple bot initialized!")

    def setup_handlers(self):
        """Setup handlers for simple bot"""

        @self.bot.message_handler(commands=['start', 'help'])
        def handle_start(message):
            self.bot.send_message(
                message.chat.id,
                "🤖 Ask me anything about Foodhub procedures!\n\n"
                "Example: 'How do I process a refund?'"
            )

        @self.bot.message_handler(commands=['clear'])
        def handle_clear(message):
            user_id = str(message.from_user.id)
            self.qa_system.clear_user_memory(user_id)
            self.bot.send_message(message.chat.id, "✅ Memory cleared!")

        @self.bot.message_handler(func=lambda m: True)
        def handle_message(message):
            chat_id = message.chat.id
            user_id = str(message.from_user.id)
            user_input = message.text

            # Send "thinking" message
            thinking_msg = self.bot.send_message(chat_id, "🤔 Thinking...")

            try:
                # Use LangChain QA with memory
                answer = self.qa_system.ask_question(
                    user_input,
                    user_id=user_id,
                    use_memory=True
                )

                # Edit message with answer
                self.bot.edit_message_text(answer[:4096], chat_id, thinking_msg.message_id)

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                self.bot.edit_message_text(error_msg, chat_id, thinking_msg.message_id)

    def run(self):
        """Start simple bot"""
        print("🚀 Simple LangChain bot running...")
        self.bot.infinity_polling()


def main():
    """Choose which bot to run"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        print("Starting Simple LangChain Bot...")
        bot = SimpleLangChainBot()
    else:
        print("Starting Advanced LangChain Bot with streaming...")
        bot = LangChainTelegramBot()

    bot.run()


if __name__ == "__main__":
    main()