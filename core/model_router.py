import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

class ModelRouter:
    """
    AI工程类：统一模型路由器。
    职责：调用DeepSeek，处理异常，记录成本。
    """
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        # 使用 __file__ 确保 cost.csv 始终保存在 core/ 目录下
        self.cost_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost.csv")
        # 如果没有成本日志文件，就创建并写个表头
        if not os.path.exists(self.cost_log_path):
            with open(self.cost_log_path, "w", encoding="utf-8") as f:
                f.write("timestamp,model,completion_tokens,prompt_tokens,total_tokens,cost_usd,latency_sec\n")

    def _log_cost(self, model, usage, latency):
        """把每一分钱都记在账上"""
        # DeepSeek的价格（2026.4.24），单位：每百万Token，你可根据实际采购价调整
        pricing = {
            # Flash版，主打轻量高效
        "deepseek-v4-flash": {"prompt": 0.14, "completion": 0.28}, # 输入: ¥1/百万, 输出: ¥2/百万，换算后约 $0.14/$0.28
            # Pro版，主打极致性能
        "deepseek-v4-pro": {"prompt": 1.68, "completion": 3.36},   # 输入: ¥12/百万, 输出: ¥24/百万，换算后约 $1.68/$3.36
        }
        pm_price = pricing.get(model, {"prompt": 0.14, "completion": 0.28})
        prompt_cost = usage.prompt_tokens * pm_price["prompt"] / 1_000_000
        completion_cost = usage.completion_tokens * pm_price["completion"] / 1_000_000
        total_cost = prompt_cost + completion_cost

        with open(self.cost_log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()},{model},{usage.completion_tokens},{usage.prompt_tokens},{usage.total_tokens},{total_cost:.6f},{latency:.2f}\n")

    def chat(self, messages, model="deepseek-v4-flash", temperature=0.7, max_tokens=1024, stream=False):
        """
        核心对话方法。
        messages格式：[{"role": "user", "content": "你好"}]
        """
        start_time = time.time()
        try:
            # 构建请求参数
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
             # 如果启用流式，则添加 stream_options
            if stream:
                request_params["stream_options"] = {"include_usage": True}

            response = self.client.chat.completions.create(**request_params)

            if stream:
                # 流式模式下，返回生成器
                def stream_wrapper():
                    collected_content = ""
                    final_usage = None   # 用于存放最终的 usage
                    
                    for chunk in response:
                        if chunk.usage:
                            final_usage = chunk.usage
                            continue
                        if chunk.choices[0].delta.content:
                            content_piece = chunk.choices[0].delta.content
                            collected_content += content_piece
                            yield content_piece
                            
                        # 流式完成后，用最终获取的usage来记录成本
                    latency = time.time() - start_time
                    if final_usage:
                        self._log_cost(model, final_usage, latency)
                        print(f"\n[Stream] 耗时 {latency:.2f}s, Token: in {final_usage.prompt_tokens} / out {final_usage.completion_tokens}")
                    else:
                        print(f"\n[Stream] 耗时 {latency:.2f}s，未能获取到usage信息进行成本记录。")
                return stream_wrapper()
                    
            else:
                # 非流式，直接解析
                latency = time.time() - start_time
                usage = response.usage
                content = response.choices[0].message.content
                
                # 记录成本
                self._log_cost(model, usage, latency)
                
                # 在控制台给你即时反馈
                print(f"[{model}] 耗时: {latency:.2f}s | Token: in {usage.prompt_tokens} / out {usage.completion_tokens} | 花费: $0.000? (已记录)")
                
                return content

        except Exception as e:
            # 网络抖动、API限流等任何异常，这里捕获并打印，然后向上抛
            print(f"[Error] 模型调用失败: {e}")
            raise e


# ----------- test -----------
if __name__ == "__main__":
    router = ModelRouter()
    
    # 测试非流式调用
    print("=== 测试非流式对话 ===")
    reply = router.chat(
        messages=[{"role": "user", "content": "用一句话解释什么是AI Agent？"}],
        model="deepseek-v4-flash",
        max_tokens=200
    )
    print("模型回复:", reply)
    
    # 测试流式调用（逐字打印）
    print("\n=== 测试流式对话 ===")
    stream_gen = router.chat(
        messages=[{"role": "user", "content": "用50字以内介绍强化学习"}],
        model="deepseek-v4-flash",
        max_tokens=100,
        stream=True
    )
    print("流式效果: ", end="", flush=True)
    for token in stream_gen:
        print(token, end="", flush=True)
    print("\n流式测试结束。")
