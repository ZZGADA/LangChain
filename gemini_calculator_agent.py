# 导入os模块，用于与操作系统交互，例如设置环境变量
import os
# 从LangChain核心库导入tool装饰器，用于轻松地将函数转换为Agent可以使用的工具
from langchain_core.tools import tool
# 导入与Google Gemini模型进行交互的聊天模型类
from langchain_google_genai import ChatGoogleGenerativeAI
# 从LangChain的agents模块导入Agent执行器和创建ReAct agent的函数
from langchain.agents import AgentExecutor, create_react_agent
# 从LangChain核心库导入用于创建和管理提示的模板类
from langchain_core.prompts import PromptTemplate
# 导入正则表达式模块，用于解析和匹配字符串
import re
# 导入json模块，用于解析JSON字符串
import json

# 设置你的Google API密钥作为环境变量。请在使用前取消注释并填入你的有效密钥。
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"


# @tool装饰器将下面的函数声明为一个可供Agent调用的工具
@tool
def calculator(expression: str) -> float:
    """
    执行单次算术运算。
    输入应该是一个遵循'数字 运算符 数字'格式的字符串。
    例如: '5 + 5' 或 '10 * 2'。
    Agent会读取这个文档字符串来理解工具的功能和使用方法。
    """
    # 清理输入字符串，去除可能由Agent错误添加的多余单引号或双引号
    expression = expression.strip("'\"")
    try:
        # 使用正则表达式来查找并分离表达式中的数字和运算符
        match = re.match(r"^\s*(-?\d+\.?\d*)\s*([+\-*\/])\s*(-?\d+\.?\d*)\s*$", expression)
        # 如果表达式格式不匹配，则返回错误信息
        if not match:
            return "错误：无效的表达式格式。请输入'数字 运算符 数字'格式的字符串。"

        # 从匹配结果中提取第一个数字、运算符和第二个数字
        a = float(match.group(1))
        op = match.group(2)
        b = float(match.group(3))

        # 根据运算符执行相应的计算
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            # 处理除以零的特殊情况
            if b == 0:
                return "错误：不能除以零"
            return a / b
        else:
            # 如果运算符无效，返回错误信息
            return "错误：无效的运算符"
    except Exception as e:
        # 捕获其他潜在错误并返回错误信息
        return f"错误: {e}"

@tool
def calculator_num(data: str) -> float:
    """
    对两个数字执行加、减、乘、除运算。
    输入应该是一个JSON格式的字符串，包含 'a', 'b', 和 'operation' 三个键。
    例如: '{"a": 234.5, "b": 11.2, "operation": "*"}'
    """
    try:
        params = json.loads(data)
        a = params['a']
        b = params['b']
        operation = params['operation']

        if operation == "+":
            return a + b
        elif operation == "-":
            return a - b
        elif operation == "*":
            return a * b
        elif operation == "/":
            if b == 0:
                return "错误：不能除以零"
            return a / b
        else:
            return "错误：无效的操作"
    except Exception as e:
        return f"错误：解析输入或计算时出错 - {e}"

# 初始化大语言模型（LLM）
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
# 此处我们选择使用gemini-1.5-flash-latest模型，它是一个速度更快、成本更低的选项
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# 创建一个列表，包含所有Agent可以使用的工具
tools = [calculator, calculator_num]

# 创建提示模板，指导Agent如何进行思考和行动
template = """
尽你所能回答以下问题。你可以使用以下工具:

{tools}

请使用以下格式:

Question: 你必须回答的输入问题
Thought: 你应该时刻思考该做什么
Action: 你要采取的行动，应该是[{tool_names}]中的一个
Action Input: 对行动的输入, 对于需要多个参数的工具，这里应该是一个包含所有参数的JSON对象。
Observation: 行动的结果
... (这个思考/行动/行动输入/观察的过程可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 对原始输入问题的最终答案

开始!

Question: {input}
Thought:{agent_scratchpad}
"""
# 从上面的模板字符串创建PromptTemplate对象
prompt = PromptTemplate.from_template(template)

# 创建一个ReAct (Reasoning and Acting) Agent
# 这个Agent结合了语言模型、工具和提示，使其能够通过思考和行动来解决问题
agent = create_react_agent(llm, tools, prompt)

# 创建Agent执行器，它负责运行Agent的决策循环
# verbose=True参数会让执行器在运行时打印出Agent的完整思考过程
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 测试Agent
# __name__ == "__main__"确保以下代码只在直接运行此脚本时执行
if __name__ == "__main__":
    # 定义要让Agent解决的问题
    question = "使用 calculator_num 工具计算234.5乘以11.2，然后再加上5等于多少？"
    # 调用Agent执行器来处理问题
    result = agent_executor.invoke({"input": question})
    # 打印最终的计算结果
    print(result)