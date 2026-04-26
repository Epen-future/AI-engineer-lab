import gradio as gr
from core.agent import Agent

agent = Agent()

def chat_with_agent(message, history):
    """
    message: 当前用户输入
    history: messages 格式的对话历史 [{"role": "user", "content": "..."}, ...]
    """
    # 1. 把当前用户消息追加到历史里
    history.append({"role": "user", "content": message})
    
    try:
        # 2. 调用Agent
        bot_response = agent.run(message)
    except Exception as e:
        bot_response = f"❌ Agent 出错：{str(e)}"
    
    # 3. 把机器人的回复也追加到历史里
    history.append({"role": "assistant", "content": bot_response})
    
    return history

# 构建界面
with gr.Blocks(title="我的 AI 助手") as demo:
    gr.Markdown("# 🤖 我的 AI 助手")
    gr.Markdown("一个能查维基百科、计算、报时的智能体。")
    
    # Gradio 6.x 中 Chatbot 已默认使用 messages 格式，无需 type 参数
    chatbot = gr.Chatbot(label="对话窗口", height=500)
    msg = gr.Textbox(label="输入你的问题", placeholder="比如：人工智能中的Transformer是什么？")
    clear = gr.Button("清空对话")
    
    # 绑定事件
    msg.submit(chat_with_agent, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )
