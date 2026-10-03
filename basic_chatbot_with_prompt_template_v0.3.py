"""This is a simple chatbot with list-based conversation history and prompt template that uses the
Azure OpenAI API to answer questions with a language model.

Make sure to set the following environment variables in a .env file:
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
AZURE_OPENAI_DEPLOYMENT_NAME=<your_azure_openai_deployment_name>
AZURE_OPENAI_API_VERSION=<your_azure_openai_api_version>
AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>"""

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

load_dotenv()

conversation_history: list[HumanMessage | AIMessage] = []

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=SecretStr(os.environ["AZURE_OPENAI_API_KEY"]),
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions based on the "
            "provided context.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | model

print(
    "Welcome to the AI Question Answering System! "
    "\nType 'exit' or 'quit' to end the conversation."
)
while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    response = chain.invoke(
        {
            "history": conversation_history,
            "question": question,
        }
    )

    conversation_history.append(HumanMessage(content=question))

    conversation_history.append(response)

    print("AI:", response.content)
    print("___________________________________________________________")
