from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def main():
    print("🤖 LangChain Gemini 实时聊天机器人")
    print("=" * 60)
    print("✨ 使用 Google Gemini 2.0 Flash 模型提供实时智能对话")
    print("🔑 需要设置 GOOGLE_API_KEY 环境变量")
    print("💡 输入 'quit', 'exit' 或 'q' 退出程序")
    print("-" * 60)
    
    # 检查API密钥
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ 错误：请设置 GOOGLE_API_KEY 环境变量")
        print("🔧 设置方法：")
        print("   export GOOGLE_API_KEY='your-google-api-key-here'")
        print("📝 获取API密钥：https://aistudio.google.com/app/apikey")
        print("\n💡 临时设置方法（仅限当前会话）：")
        api_key = input("请直接输入您的 Google API Key (按回车跳过): ").strip()
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            print("✅ API Key 已临时设置")
        else:
            print("❌ 未设置API密钥，程序退出")
            return
    
    try:
        global llm
        # 初始化Gemini模型 - 使用正确的模型名称
        print("🔄 正在初始化 Gemini 模型...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # 使用最新的Gemini 2.0 Flash模型
            temperature=0.7,               # 创造性设置
            max_tokens=2000,              # 增加回复长度
        )
        
        # 测试模型连接
        print("🧪 测试模型连接...")
        test_response = llm.invoke([HumanMessage(content="Hello, can you respond in Chinese?")])
        if not test_response.content:
            raise Exception("模型响应为空")
        
        # 创建状态图
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        
        # 编译图
        graph = graph_builder.compile()
        
        print("✅ Gemini 2.0 Flash 模型初始化成功！")
        print("🎉 您现在可以开始与 AI 进行实时对话了！\n")
        
        # 显示模型信息
        print(f"📋 模型信息:")
        print(f"   • 模型: Gemini 2.0 Flash Experimental")
        print(f"   • 温度: 0.7 (创造性)")
        print(f"   • 最大令牌: 2000")
        print(f"   • 支持: 多语言实时对话\n")
        
        # 添加欢迎消息
        print("🤖 助手: 你好！我是基于 Google Gemini 2.0 的 AI 助手。我可以进行实时对话，帮助你回答问题、进行创意讨论、解决问题等。请随时告诉我你想聊什么！\n")
        
        conversation_count = 0
        
        while True:
            try:
                user_input = input("👤 你: ")
                
                if user_input.lower() in ["quit", "exit", "q", "退出", "再见"]:
                    print("👋 谢谢使用 Gemini 实时聊天机器人！再见！")
                    break
                
                if not user_input.strip():
                    print("💭 请输入一些内容...")
                    continue
                
                conversation_count += 1
                print(f"🤖 助手 (第{conversation_count}轮): ", end="", flush=True)
                
                # 实时处理对话
                try:
                    response = ""
                    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}):
                        for value in event.values():
                            current_response = value["messages"][-1].content
                            if current_response != response:
                                # 实时打印新内容
                                new_content = current_response[len(response):]
                                print(new_content, end="", flush=True)
                                response = current_response
                    
                    print("\n")  # 换行
                    
                except Exception as e:
                    print(f"\n❌ 对话处理失败: {e}")
                    print("💡 请检查网络连接或重试")
                        
            except KeyboardInterrupt:
                print("\n👋 程序被中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 输入处理出错: {e}")
                print("💡 请重试")
                
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("\n🔧 可能的解决方案：")
        print("1. 检查 GOOGLE_API_KEY 是否正确设置")
        print("2. 确认网络连接正常")
        print("3. 验证 API 密钥是否有效且有足够配额")
        print("4. 检查防火墙设置")
        print("5. 重新安装依赖：pip install -r requirements.txt")
        
        # 提供备用方案
        print("\n🆘 如果问题持续，您可以：")
        print("1. 尝试使用本地示例：python langchain_local_example.py")
        print("2. 检查 Google AI Studio 状态：https://aistudio.google.com/")


if __name__ == "__main__":
    main() 