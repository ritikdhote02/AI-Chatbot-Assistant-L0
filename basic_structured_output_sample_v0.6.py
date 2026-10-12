"""Demonstrate Azure OpenAI structured output using a Pydantic model.

The example extracts a person's name, age, and occupation from text, then
prints the structured result and its dictionary representation.

Make sure to set the following environment variables in a .env file:
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
AZURE_OPENAI_DEPLOYMENT_NAME=<your_azure_openai_deployment_name>
AZURE_OPENAI_API_VERSION=<your_azure_openai_api_version>
AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>"""

import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

# 1. Load environment variables
load_dotenv()


# 2. Configure the Azure OpenAI model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


# 3. Define the expected output schema
class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or occupation")


# 4. Configure structured output
structured_model = model.with_structured_output(Person)


# 5. Provide input text

# text = """
# I am Ritik, my age is twenty-three,
# and I am a software engineer.
# """

text = """
My name is Ritik, I am 23 years old,
and I work as a software engineer.
"""

# 6. Invoke the model
result = structured_model.invoke(text)


# 7. Access the structured result
print("Name:", result.name)
print("Age:", result.age)
print("Occupation:", result.occupation)


# 8. Convert to a dictionary
print("Dictionary:", result.model_dump())
