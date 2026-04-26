import gradio as gr
from core.agent import Agent,Step

agent = Agent()

def chat_with_agent(message, history):
    # 添加用户消息
    history.append({"role": "user", "content": message})
    # 占位消息
    history.append({"role": "assistant", "content": "🤔 正在思考..."})
    yield history

    steps = []
    answer_buffer = ""  # 用于累积流式 token

    try:
        for item in agent.run_stream(message, history=history[:-1]):
            if isinstance(item, Step):
                steps.append(item.message)
                # 在占位区显示最新的思考步骤
                display = "\n".join(steps) + "\n\n⏳ 生成中..."
                history[-1] = {"role": "assistant", "content": display}
                yield history
            else:  # str token
                answer_buffer += item
                # 实时更新为当前累积的答案
                history[-1] = {"role": "assistant", "content": answer_buffer}
                yield history
        
        # 流式结束，构造最终展示：思考过程 + 最终答案
        if steps:
            final_display = "**💭 思考过程：**\n> " + "\n> ".join(steps) + "\n\n**📝 回答：**\n" + answer_buffer
        else:
            final_display = answer_buffer
        history[-1] = {"role": "assistant", "content": final_display}
        yield history
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"❌ 出错：{str(e)}"}
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
