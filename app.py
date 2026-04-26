import gradio as gr
from core.agent import Agent

agent = Agent()

def chat_with_agent(message, history):
    # 添加用户消息
    history.append({"role": "user", "content": message})
    # 先占个位，显示初始状态
    history.append({"role": "assistant", "content": "🤔 正在思考，请稍候..."})
    yield history

    status_log = []         # 收集思考过程
    def update_status(msg):
        status_log.append(msg)
        # 实时在界面显示最新状态，但不覆盖最终答案
        combined = "\n".join(status_log)
        history[-1] = {"role": "assistant", "content": f"🤔 **思考中...**\n\n> {combined}"}
        yield history

    try:
        final_answer = agent.run(
            user_query=message,
            history=history[:-1],
            status_callback=update_status
        )
    except Exception as e:
        final_answer = f"❌ Agent 出错：{str(e)}"

    # 最终答案：思考过程折叠，答案放在下面
    process_text = "\n> ".join(status_log) if status_log else "无思考过程"
    formatted_answer = (
        f"**💭 思考过程：**\n"
        f"> {process_text}\n\n"
        f"**📝 最终回答：**\n"
        f"{final_answer}"
    )
    history[-1] = {"role": "assistant", "content": formatted_answer}
    yield history

# 构建界面
with gr.Blocks(title="我的 AI 助手") as demo:
    gr.Markdown("# 🤖 我的 AI 助手")
    gr.Markdown("一个能查百科、计算、报时的智能体。")
    
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
