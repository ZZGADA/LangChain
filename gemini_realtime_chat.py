#!/usr/bin/env python3
"""
Gemini 实时聊天机器人 - 简化版本
直接使用 Google Gemini API，提供最快的实时对话体验
"""

import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class GeminiRealTimeChat:
    def __init__(self):
        self.llm = None
        self.conversation_history = []
        self.conversation_count = 0
        
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
    
    def initialize_model(self):
        """初始化Gemini模型"""
        try:
            print("🔄 正在初始化 Gemini 模型...")
            
            # 尝试不同的模型名称
            models_to_try = [
                "gemini-2.0-flash-exp",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-pro"
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"🧪 尝试模型: {model_name}")
                    self.llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0.7,
                        max_tokens=2000,
                    )
                    
                    # 测试模型连接
                    test_response = self.llm.invoke([HumanMessage(content="Hello")])
                    if test_response and test_response.content:
                        print(f"✅ 成功连接到 {model_name}")
                        return True
                        
                except Exception as e:
                    print(f"❌ {model_name} 连接失败: {str(e)[:100]}...")
                    continue
            
            print("❌ 所有模型都连接失败")
            return False
            
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            return False
    
    def add_system_message(self):
        """添加系统消息"""
        system_msg = SystemMessage(content="""你是一个友好、智能的AI助手。请用中文回答问题，保持对话自然流畅。
你可以帮助用户：
- 回答各种问题
- 进行创意讨论
- 解决问题
- 提供建议和指导
请保持回答简洁明了，但又足够详细。""")
        self.conversation_history.append(system_msg)
    
    def chat_with_gemini(self, user_input):
        """与Gemini进行对话"""
        try:
            # 添加用户消息到历史记录
            self.conversation_history.append(HumanMessage(content=user_input))
            
            # 保持对话历史在合理范围内（最近10轮对话）
            if len(self.conversation_history) > 21:  # 1系统消息 + 20条对话消息
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
            
            # 获取AI回复
            response = self.llm.invoke(self.conversation_history)
            
            if response and response.content:
                # 添加AI回复到历史记录
                self.conversation_history.append(AIMessage(content=response.content))
                return response.content
            else:
                return "抱歉，我没有收到有效的回复。请重试。"
                
        except Exception as e:
            return f"对话出错: {e}\n请检查网络连接或重试。"
    
    def start_chat(self):
        """开始聊天"""
        print("🤖 Gemini 实时聊天机器人")
        print("=" * 60)
        print("✨ 基于 Google Gemini 的实时智能对话")
        print("💡 输入 'quit', 'exit', 'q' 或 '退出' 结束对话")
        print("-" * 60)
        
        # 设置API密钥
        if not self.setup_api_key():
            print("❌ API密钥设置失败，程序退出")
            return
        
        # 初始化模型
        if not self.initialize_model():
            print("❌ 模型初始化失败，程序退出")
            return
        
        # 添加系统消息
        self.add_system_message()
        
        print("🎉 Gemini 聊天机器人已就绪！")
        print("📋 功能特性：")
        print("   • 实时对话响应")
        print("   • 记住对话上下文")
        print("   • 支持中文对话")
        print("   • 智能问答与创意讨论")
        print()
        
        # 欢迎消息
        print("🤖 Gemini: 你好！我是 Google Gemini AI 助手。我可以帮你回答问题、进行讨论、解决问题等。请告诉我你想聊什么吧！\n")
        
        # 开始对话循环
        while True:
            try:
                user_input = input("👤 你: ").strip()
                
                if not user_input:
                    print("💭 请输入一些内容...")
                    continue
                
                if user_input.lower() in ["quit", "exit", "q", "退出", "再见", "bye"]:
                    print("👋 谢谢使用 Gemini 聊天机器人！再见！")
                    break
                
                # 特殊命令
                if user_input.lower() in ["clear", "清空", "重置"]:
                    self.conversation_history = []
                    self.add_system_message()
                    self.conversation_count = 0
                    print("🔄 对话历史已清空\n")
                    continue
                
                if user_input.lower() in ["help", "帮助"]:
                    print("📚 可用命令：")
                    print("   • clear/清空 - 清空对话历史")
                    print("   • help/帮助 - 显示此帮助信息")
                    print("   • quit/退出 - 结束对话\n")
                    continue
                
                self.conversation_count += 1
                print(f"🤖 Gemini (第{self.conversation_count}轮): ", end="", flush=True)
                
                # 获取AI回复
                response = self.chat_with_gemini(user_input)
                
                # 实时打印回复（模拟打字效果）
                import time
                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(0.01)  # 轻微延迟，模拟打字效果
                
                print("\n")  # 换行
                
            except KeyboardInterrupt:
                print("\n👋 程序被中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                print("💡 请重试")


def main():
    """主函数"""
    try:
        chat_bot = GeminiRealTimeChat()
        chat_bot.start_chat()
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        print("💡 请检查您的环境配置")


if __name__ == "__main__":
    main() 