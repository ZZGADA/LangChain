#!/usr/bin/env python3
"""
LangChain Gemini 聊天机器人 - 完整版
使用 LangChain + LangGraph 框架构建的聊天机器人
展示框架的完整功能和最佳实践
"""

import os
import sys
import time
from typing import Annotated, List, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class ChatState(TypedDict):
    """聊天状态定义"""
    messages: Annotated[List, add_messages]
    user_info: Optional[dict]
    conversation_count: int
    last_activity: str


class LangChainGeminiBot:
    """基于 LangChain 的 Gemini 聊天机器人"""
    
    def __init__(self):
        self.llm = None
        self.graph = None
        self.memory = MemorySaver()  # 内存保存器
        self.thread_config = {"configurable": {"thread_id": "main_conversation"}}
        
    def setup_api_key(self):
        """设置API密钥"""
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌ 未检测到 GOOGLE_API_KEY 环境变量")
            print("\n🔑 获取API密钥步骤：")
            print("1. 访问：https://aistudio.google.com/app/apikey")
            print("2. 登录Google账号")
            print("3. 创建新的API密钥")
            print("4. 复制密钥")
            
            print("\n💡 请选择设置方式：")
            print("1. 临时设置（仅本次会话有效）")
            print("2. 永久设置（推荐）")
            
            choice = input("请选择 (1/2): ").strip()
            
            if choice == "1":
                api_key = input("\n请输入您的 Google API Key: ").strip()
                if api_key:
                    os.environ["GOOGLE_API_KEY"] = api_key
                    print("✅ API Key 已临时设置")
                    return True
                else:
                    print("❌ 未输入API密钥")
                    return False
            elif choice == "2":
                print("\n🔧 永久设置方法：")
                print("在终端中运行以下命令：")
                print("export GOOGLE_API_KEY='your-api-key-here'")
                print("\n或者将上述命令添加到 ~/.bashrc 或 ~/.zshrc 文件中")
                return False
            else:
                print("❌ 无效选择")
                return False
        
        return True
    
    def test_network_connection(self):
        """测试网络连接"""
        print("🌐 正在测试网络连接...")
        try:
            import urllib.request
            import socket
            
            # 测试基本网络连接
            socket.setdefaulttimeout(10)
            urllib.request.urlopen('https://www.google.com', timeout=10)
            print("✅ 网络连接正常")
            
            # 测试 Google AI API 端点
            try:
                urllib.request.urlopen('https://generativelanguage.googleapis.com', timeout=10)
                print("✅ Google AI API 端点可达")
                return True
            except Exception as e:
                print(f"⚠️  Google AI API 端点连接异常: {e}")
                print("💡 可能是网络防火墙或代理问题")
                return True  # 允许继续尝试
                
        except Exception as e:
            print(f"❌ 网络连接失败: {e}")
            print("💡 请检查网络连接或代理设置")
            return False
    
    def initialize_llm(self):
        """初始化语言模型"""
        try:
            print("🔄 正在初始化 LangChain Gemini 模型...")
            
            # 先测试网络连接
            # if not self.test_network_connection():
            #     return False

            # 尝试不同的模型，优先使用稳定版本
            models_to_try = [
                ("gemini-2.5-pro", "最新版本"),
                ("gemini-1.5-flash", "稳定版本，推荐使用"),
                ("gemini-1.5-pro", "高级版本"),
                ("gemini-pro", "经典版本"),
                ("gemini-2.0-flash-exp", "实验版本"),
            ]

            for model_name, description in models_to_try:
                try:
                    print(f"🧪 尝试模型: {model_name} ({description})")
                    
                    # 创建模型实例，增加超时设置
                    self.llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0.7,
                        max_tokens=2000,
                        timeout=30,  # 增加超时时间
                        max_retries=2,  # 减少重试次数
                    )
                    
                    # 测试模型连接，使用简单的消息
                    print("   🔄 测试模型连接...")
                    test_response = self.llm.invoke(
                        [HumanMessage(content="Hi")], 
                        config={"timeout": 15}  # 设置调用超时
                    )
                    
                    if test_response and test_response.content:
                        print(f"✅ 成功连接到 {model_name}")
                        print(f"   📝 测试响应: {test_response.content[:50]}...")
                        return True
                        
                except KeyboardInterrupt:
                    print("\n⚠️  用户中断了连接测试")
                    return False
                except Exception as e:
                    error_msg = str(e)
                    if "ServiceUnavailable" in error_msg or "UNAVAILABLE" in error_msg:
                        print(f"   ❌ {model_name} 服务不可用，可能是网络问题")
                    elif "PERMISSION_DENIED" in error_msg:
                        print(f"   ❌ {model_name} 权限被拒绝，请检查API密钥")
                    elif "NOT_FOUND" in error_msg:
                        print(f"   ❌ {model_name} 模型不存在或不可用")
                    else:
                        print(f"   ❌ {model_name} 连接失败: {error_msg[:100]}...")
                    
                    # 短暂等待后尝试下一个模型
                    time.sleep(1)
                    continue
            
            print("❌ 所有模型都连接失败")
            print("\n🔧 可能的解决方案：")
            print("1. 检查网络连接是否正常")
            print("2. 确认API密钥是否正确且有效")
            print("3. 检查是否有防火墙或代理阻止连接")
            print("4. 稍后重试，可能是服务暂时不可用")
            print("5. 尝试使用实时聊天版本：python gemini_realtime_chat.py")
            return False
            
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            return False
    
    def create_prompt_template(self):
        """创建提示模板"""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""你是一个友好、智能的AI助手，名字叫 LangChain Gemini Bot。

你的特点：
- 使用中文进行自然流畅的对话
- 能够记住对话历史和上下文
- 提供准确、有用的信息和建议
- 保持专业但友好的语调
- 可以进行创意讨论和问题解决

当前时间：{current_time}

请根据用户的问题提供有帮助的回答。"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def chatbot_node(self, state: ChatState):
        """聊天机器人节点 - LangGraph 的核心处理单元"""
        try:
            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 创建提示模板
            prompt = self.create_prompt_template()
            
            # 创建处理链
            chain = (
                {
                    "messages": lambda x: x["messages"],
                    "current_time": lambda x: current_time,
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # 调用处理链，增加超时控制
            response = chain.invoke(state, config={"timeout": 30})
            
            # 更新状态
            return {
                "messages": [AIMessage(content=response)],
                "conversation_count": state.get("conversation_count", 0) + 1,
                "last_activity": current_time,
            }
            
        except Exception as e:
            error_msg = f"处理消息时出错: {e}"
            print(f"⚠️  {error_msg}")
            return {
                "messages": [AIMessage(content="抱歉，我遇到了一些技术问题。请重试或检查网络连接。")],
                "conversation_count": state.get("conversation_count", 0),
                "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
    
    def should_continue(self, state: ChatState):
        """决定是否继续处理"""
        # 这里可以添加复杂的逻辑判断
        # 例如：检查是否需要调用工具、是否需要额外处理等
        return "continue"

    """
   1. 创建状态图 (StateGraph)：初始化一个 StateGraph 对象，并指定 ChatState 。ChatState
      用来在工作流的每一步之间传递数据（如消息历史、用户信息等）。
   2. 添加节点 (Node)：它添加了一个名为 "chatbot" 的核心节点，这个节点关联到 self.chatbot_node 方法。节点是图中的基本处理单元，chatbot_node
      负责调用大语言模型并获取回复。
   3. 定义流程边 (Edge)：
       * workflow.add_edge(START, "chatbot"): 这条边定义了工作流的入口。当图开始执行时，会首先进入 "chatbot" 节点。
       * workflow.add_edge("chatbot", END): 这条边定义了 "chatbot" 节点执行完毕后，工作流就结束了。
   4. 编译图 (Compile)：最后，它调用 workflow.compile(checkpointer=self.memory) 来将定义好的节点和边编译成一个可执行的图。关键在于
      checkpointer=self.memory，它为图添加了记忆功能，使得每次调用的状态（比如对话历史）都能被保存和恢复。

    
    """
    def build_graph(self):
        """构建 LangGraph 状态图"""
        try:
            print("🔨 正在构建 LangGraph 状态图...")
            
            # 创建状态图
            workflow = StateGraph(ChatState)
            
            # 添加节点
            workflow.add_node("chatbot", self.chatbot_node)
            
            # 添加边
            workflow.add_edge(START, "chatbot")
            workflow.add_edge("chatbot", END)
            
            # 编译图（带内存）
            self.graph = workflow.compile(checkpointer=self.memory)
            
            print("✅ LangGraph 状态图构建成功")
            return True
            
        except Exception as e:
            print(f"❌ 状态图构建失败: {e}")
            return False
    
    def process_user_input(self, user_input: str):
        """处理用户输入"""
        try:
            # 创建初始状态
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "conversation_count": 0,
                "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # 使用图处理，增加超时控制
            result = self.graph.invoke(
                initial_state, 
                config={
                    **self.thread_config,
                    "timeout": 45  # 增加处理超时时间
                }
            )
            
            # 获取最后一条AI消息
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "抱歉，我没有生成有效的回复。"
                
        except Exception as e:
            error_msg = f"处理输入时出错: {e}"
            print(f"⚠️  {error_msg}")
            return "抱歉，我遇到了处理问题。请重试或检查网络连接。"
    
    def get_conversation_stats(self):
        """获取对话统计信息"""
        try:
            # 从内存中获取当前状态
            current_state = self.graph.get_state(self.thread_config)
            if current_state and current_state.values:
                return {
                    "conversation_count": current_state.values.get("conversation_count", 0),
                    "last_activity": current_state.values.get("last_activity", "未知"),
                    "message_count": len(current_state.values.get("messages", [])),
                }
        except:
            pass
        return {"conversation_count": 0, "last_activity": "未知", "message_count": 0}
    
    def clear_memory(self):
        """清空对话记忆"""
        try:
            # 重新创建内存保存器
            self.memory = MemorySaver()
            # 重新构建图
            workflow = StateGraph(ChatState)
            workflow.add_node("chatbot", self.chatbot_node)
            workflow.add_edge(START, "chatbot")
            workflow.add_edge("chatbot", END)
            self.graph = workflow.compile(checkpointer=self.memory)
            return True
        except Exception as e:
            print(f"清空记忆失败: {e}")
            return False
    
    def start_chat(self):
        """开始聊天"""
        print("🤖 LangChain Gemini 聊天机器人")
        print("=" * 60)
        print("✨ 基于 LangChain + LangGraph 框架的智能对话")
        print("🧠 支持对话记忆和上下文理解")
        print("💡 输入 'quit', 'exit', 'q' 或 '退出' 结束对话")
        print("-" * 60)
        
        # 设置API密钥
        if not self.setup_api_key():
            print("❌ API密钥设置失败，程序退出")
            return
        
        # 初始化模型
        if not self.initialize_llm():
            print("❌ 模型初始化失败，程序退出")
            print("\n💡 备用方案：")
            print("1. 尝试实时聊天版本：python gemini_realtime_chat.py")
            print("2. 使用本地示例：python langchain_local_example.py")
            return
        
        # 构建图
        if not self.build_graph():
            print("❌ LangGraph 构建失败，程序退出")
            return
        
        print("🎉 LangChain Gemini 聊天机器人已就绪！")
        print("📋 框架特性：")
        print("   • LangChain 组件化架构")
        print("   • LangGraph 状态管理")
        print("   • 对话记忆功能")
        print("   • 网络异常处理")
        print("   • 可扩展的工作流")
        print()
        
        # 欢迎消息
        print("🤖 LangChain Gemini: 你好！我是基于 LangChain 框架的 Gemini AI 助手。")
        print("我可以记住我们的对话历史，进行更智能的交互。请告诉我你想聊什么吧！\n")
        
        # 主对话循环
        while True:
            try:
                user_input = input("👤 你: ").strip()
                
                if not user_input:
                    print("💭 请输入一些内容...")
                    continue
                
                if user_input.lower() in ["quit", "exit", "q", "退出", "再见", "bye"]:
                    stats = self.get_conversation_stats()
                    print(f"📊 对话统计: {stats['conversation_count']} 轮对话, {stats['message_count']} 条消息")
                    print("👋 谢谢使用 LangChain Gemini 聊天机器人！再见！")
                    break
                
                # 特殊命令
                if user_input.lower() in ["clear", "清空", "重置"]:
                    if self.clear_memory():
                        print("🔄 对话历史已清空\n")
                    else:
                        print("❌ 清空失败\n")
                    continue
                
                if user_input.lower() in ["stats", "统计", "状态"]:
                    stats = self.get_conversation_stats()
                    print("📊 对话统计信息：")
                    print(f"   • 对话轮数: {stats['conversation_count']}")
                    print(f"   • 消息总数: {stats['message_count']}")
                    print(f"   • 最后活动: {stats['last_activity']}\n")
                    continue
                
                if user_input.lower() in ["help", "帮助"]:
                    print("📚 可用命令：")
                    print("   • clear/清空 - 清空对话历史")
                    print("   • stats/统计 - 显示对话统计")
                    print("   • help/帮助 - 显示此帮助信息")
                    print("   • quit/退出 - 结束对话\n")
                    continue
                
                # 获取对话统计
                stats = self.get_conversation_stats()
                print(f"🤖 LangChain Gemini (第{stats['conversation_count']+1}轮): ", end="", flush=True)
                
                # 处理用户输入
                response = self.process_user_input(user_input)
                
                # 实时打印响应（模拟打字效果）
                import time
                for char in response:
                    print(char, end="", flush=True)
                    # time.sleep(0.01)
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\n👋 程序被中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                print("💡 请重试")


def main():
    """主函数"""
    try:
        bot = LangChainGeminiBot()
        bot.start_chat()
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        print("💡 请检查您的环境配置")
        print("🔧 备用方案：")
        print("1. python gemini_realtime_chat.py  # 实时聊天版本")
        print("2. python langchain_local_example.py  # 本地示例版本")


if __name__ == "__main__":
    main() 