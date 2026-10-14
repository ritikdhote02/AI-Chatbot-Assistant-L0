"""This is a simple chatbot that uses the Azure OpenAI API to answer questions
with a language model.

Make sure to set the following environment variables in a .env file:
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
AZURE_OPENAI_DEPLOYMENT_NAME=<your_azure_openai_deployment_name>
AZURE_OPENAI_API_VERSION=<your_azure_openai_api_version>
AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>"""

import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=SecretStr(os.environ["AZURE_OPENAI_API_KEY"]),
)

# response = model.invoke("What is artificial intelligence?")
print("Welcome to the AI Question Answering System!")
question = input("Enter your question: ")
response = model.invoke(question)

print("AI: ", response.content)
