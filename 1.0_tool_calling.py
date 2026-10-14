
import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI


# 1. Load environment variables from .env
load_dotenv()


# 2. Initialize the Azure OpenAI model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


# 3. Define a tool that the LLM can request
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


# 4. Register the available tools with the model
tools = [multiply]

tools_by_name = {
    tool.name: tool
    for tool in tools
}

model_with_tools = model.bind_tools(tools)


# 5. Send a question to the model
question = "What is 125 multiplied by 48? Use the calculator tool."

messages = [HumanMessage(content=question)]

response = model_with_tools.invoke(messages)

# Save the model's response, including any tool-call request
messages.append(response)


# 6. Execute requested tools and return their results to the model
while response.tool_calls:

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"\nTool requested: {tool_name}")
        print(f"Tool arguments: {tool_args}")

        # Find and execute the requested tool
        selected_tool = tools_by_name.get(tool_name)

        if selected_tool is None:
            raise ValueError(f"Unknown tool requested: {tool_name}")

        result = selected_tool.invoke(tool_args)

        print(f"Tool result: {result}")

        # Send the result back with its matching tool-call ID
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
            )
        )

    # Ask the model to continue using the tool results
    response = model_with_tools.invoke(messages)
    messages.append(response)


# 7. Display the final response
print("\nFinal answer:", response.content)
