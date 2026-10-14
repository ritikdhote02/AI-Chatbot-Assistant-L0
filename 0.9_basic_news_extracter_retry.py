"""Extract structured news details from an article using Azure OpenAI.

The script classifies the article and returns a headline, source, summary,
sentiment, and key entities using a Pydantic schema with try except blocks and retry logic, then prints the result
as JSON.

Make sure to set the following environment variables in a .env file:
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
AZURE_OPENAI_DEPLOYMENT_NAME=<your_azure_openai_deployment_name>
AZURE_OPENAI_API_VERSION=<your_azure_openai_api_version>
AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Configure Azure OpenAI
# --------------------------------------------------

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


# --------------------------------------------------
# 3. Define the news extraction schema
# --------------------------------------------------


class NewsExtraction(BaseModel):
    headline: str = Field(
        description="A concise headline that accurately represents the article."
    )

    source_name: str = Field(description="The name of the news source or publication.")

    category: Literal[
        "Technology",
        "Business",
        "Politics",
        "Science",
        "Health",
        "Environment",
        "World",
        "Sports",
        "Other",
    ] = Field(description="The single best-fitting category for the article.")

    summary: str = Field(
        description="A brief summary of the main facts stated in the article."
    )

    sentiment: Literal[
        "Positive",
        "Negative",
        "Neutral",
        "Mixed",
        "Unclear",
    ] = Field(
        description=(
            "The overall tone of the article. "
            "Use Neutral for factual reporting without a clear emotional tone."
        )
    )

    key_entities: list[str] = Field(
        description=(
            "Names of relevant people, organizations, and places "
            "explicitly mentioned in the article."
        )
    )


# --------------------------------------------------
# 4. Configure structured output
# --------------------------------------------------

structured_model = model.with_structured_output(NewsExtraction)


# --------------------------------------------------
# 5. Define the extraction prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a careful news analysis assistant.

            Extract information only from the supplied article.
            Do not invent facts, names, dates, or numbers.
            Do not use outside knowledge to fill missing information.
            Write a concise headline and summary.
            Choose only the allowed category and sentiment values.
            Include only relevant entities explicitly mentioned
            in the article.
            If information is missing, do not guess.
            """,
        ),
        (
            "human",
            "Extract structured information from this article:\n\n{article}",
        ),
    ]
)


# --------------------------------------------------
# 6. Build the chain and retry chain
# --------------------------------------------------

chain = prompt | structured_model

retrying_chain = chain.with_retry(
    retry_if_exception_type=(
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    ),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

# --------------------------------------------------
# 7. Test with a sample article
# --------------------------------------------------

article_text = """
Leader of the Opposition Rahul Gandhi and nine INDIA Bloc leaders have filed a writ petition in the Supreme Court challenging the Election Commission of India's (ECI) Special Intensive Revision (SIR) of electoral rolls. Apart from Congress leaders Gandhi and Mallikarjun Kharge, Saugata Roy (Trinamool Congress-Mamata Banerjee), Akhilesh Yadav (Samajwadi Party), Tejashwi Prasad Yadav (RJD), Sanjay Raut (Shiv Sena UBT), MA Baby (CPI(M)), D Raja (CPI), Mahua Majhi (JMM), Supriya Sule (NCP Sharad Pawar) and ET Muhammed Basheer (IUML) are the other petitioners.
The petition arrays the Election Commission of India, Chief Election Commissioner Gyanesh Kumar and other Election Commissioners Sukhbir Singh Sandhu and Vivek Joshi as the respondents. The petition comes amid the Congress-led Opposition's protests against the SIR exercise and seeking the resignation of Chief Election Commissioner Gyanesh Kumar over the alleged irregularities in the voters' roll revision.
"""


def validate_article(article: str) -> str:
    if not article or not article.strip():
        raise ValueError("Article cannot be empty.")

    if len(article) > 20_000:
        raise ValueError("Article is too long.")

    return article.strip()


try:
    article_text = validate_article(article_text)
    # --------------------------------------------------
    # 8. Run extraction
    # --------------------------------------------------

    result = retrying_chain.invoke(
        {
            "article": article_text,
        }
    )

    # --------------------------------------------------
    # 9. Display the result
    # --------------------------------------------------

    print("\nNEWS EXTRACTION RESULT")
    print("----------------------")

    print("Headline:", result.headline)
    print("Category:", result.category)
    print("Summary:", result.summary)
    print("Sentiment:", result.sentiment)
    print("Key entities:", result.key_entities)

    # --------------------------------------------------
    # 10. Export as JSON
    # --------------------------------------------------

    print("\nJSON OUTPUT")
    print(result.model_dump_json(indent=2))

except OutputParserException as e:
    print("The model output could not be parsed.")
    print("Error details:", e)

except AuthenticationError:
    print("Authentication failed. Check your Azure OpenAI credentials.")

except RateLimitError:
    print("The API rate limit was reached. Try again later.")

except APITimeoutError:
    print("The Azure OpenAI request timed out.")

except APIConnectionError:
    print("Could not connect to Azure OpenAI. Check network connectivity.")

except BadRequestError:
    print("Azure OpenAI rejected the request. Check the request configuration.")

except ValidationError:
    print("The extracted data did not pass Pydantic validation.")

except APIStatusError as e:
    print(f"Azure OpenAI returned an HTTP error (status {e.status_code}).")

except Exception as e:
    print("Error details:", e)
