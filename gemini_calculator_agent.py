
import os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# Set your Google API key as an environment variable
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

import re

@tool
def calculator(expression: str) -> float:
    """
    Performs a single arithmetic operation.
    The input should be a string in the format: 'number operator number'
    For example: '5 + 5' or '10 * 2'.
    """
    expression = expression.strip("'\"")
    try:
        # Use regex to find numbers and the operator
        match = re.match(r"^\s*(-?\d+\.?\d*)\s*([+\-*\/])\s*(-?\d+\.?\d*)\s*$", expression)
        if not match:
            return "Error: Invalid expression format. Please use 'number operator number'."

        a = float(match.group(1))
        op = match.group(2)
        b = float(match.group(3))

        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            if b == 0:
                return "Error: Division by zero"
            return a / b
        else:
            return "Error: Invalid operator"
    except Exception as e:
        return f"Error: {e}"

# Initialize the LLM
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
# Use gemini-1.5-flash-latest which is a faster and cheaper model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")


tools = [calculator]

# Create the prompt template
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(template)

# Create the agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Test the agent
if __name__ == "__main__":
    question = "What is 234.5 * 11.2, and then add 5?"
    result = agent_executor.invoke({"input": question})
    print(result)
