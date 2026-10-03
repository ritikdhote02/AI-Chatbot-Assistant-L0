"""This is a simple chatbot with simple memory implementation using list that uses the Azure OpenAI API to answer questions using a language model.
Make sure to set the following environment variables in a .env file:
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
AZURE_OPENAI_DEPLOYMENT_NAME=<your_azure_openai_deployment_name>
AZURE_OPENAI_API_VERSION=<your_azure_openai_api_version>
AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

conversation_history = []

print("Welcome to the AI Question Answering System! \nType 'exit' or 'quit' to end the conversation.")
while True:
    question = input("Enter your question: ")
    if question.lower() in ["exit", "quit"]:
        print("Exiting the AI Question Answering System. Goodbye!")
        break
    conversation_history.append(HumanMessage(content=question))
    response = model.invoke(conversation_history)
    conversation_history.append(AIMessage(content=response.content))
    print("AI: ", response.content)
    print("___________________________________________________________")

# print("___________________________________________________________")
# print ("Conversation History:")
# for message in conversation_history:
#     if isinstance(message, HumanMessage):
#         print("User: ", message.content)
#     elif isinstance(message, AIMessage):
#         print("AI: ", message.content)
#     print("___________________________________________________________")
# print("___________________________________________________________")
