"""This is a simple chatbot with list-based conversation history with the basic session concept that uses the
Azure OpenAI API to answer questions with a language model.

Make sure to set the following environment variables in a .env file:
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
AZURE_OPENAI_DEPLOYMENT_NAME=<your_azure_openai_deployment_name>
AZURE_OPENAI_API_VERSION=<your_azure_openai_api_version>
AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>"""

import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import AzureChatOpenAI

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Create Azure OpenAI model
# --------------------------------------------------

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


# --------------------------------------------------
# 3. Create prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. "
            "Answer the user's questions clearly and accurately.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


# --------------------------------------------------
# 4. Create basic chain
# --------------------------------------------------

chain = prompt | model


# --------------------------------------------------
# 5. Create message history store
# --------------------------------------------------

store = {}


def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# --------------------------------------------------
# 6. Add message history to our chain
# --------------------------------------------------

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# --------------------------------------------------
# 7. Chat loop
# --------------------------------------------------

session_id = "user_001"

print(
    "Welcome to the AI Question Answering System! "
    "\nType 'exit' or 'quit' to end the conversation."
)

while True:

    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        print("Exiting the AI Question Answering System. Goodbye!")
        break

    response = chain_with_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )

    print("AI:", response.content)
    print("___________________________________________________________")

# print("Conversation history:")
# for message in get_session_history(session_id).messages:
#     print(f"{message.type.capitalize()}: {message.content}")    
