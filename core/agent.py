import json
from datetime import datetime
from core.model_router import ModelRouter
import requests
from bs4 import BeautifulSoup

class Step:
    def __init__(self, message):
        self.message = message

# ---------- 1. 定义工具（你的Agent的双手）----------
def calculator(expression: str) -> str:
    """
    安全计算数学表达式，如 "12 + 3 * 4"
    """
    try:
        # 只允许数字、四则运算、括号和小数点，防止代码注入
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式中包含非法字符"
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错：{str(e)}"

def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前时间，默认格式为年-月-日 时:分:秒"""
    return datetime.now().strftime(format)

def search_baike(query: str) -> str:
    """
    使用百度百科页面获取知识摘要，带降级解析。
    """
    # 关键词净化：去掉括号及里面的内容，只保留核心词
    import re
    clean_query = re.sub(r'\(.*?\)', '', query).strip()
    if not clean_query:
        clean_query = query
    
    try:
        url = f"https://baike.baidu.com/item/{requests.utils.quote(clean_query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return f"百度百科查询失败，状态码：{resp.status_code}"
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 第一优先：标准的摘要div
        summary_div = soup.find('div', class_='lemma-summary')
        if summary_div:
            text = summary_div.get_text(separator='\n', strip=True)
            return f"百度百科词条 '{clean_query}'：\n{text}\n链接：{url}"
        
        # 第二优先：页面meta描述
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc['content'].strip()
            return f"百度百科词条 '{clean_query}' 简介：\n{desc}\n链接：{url}"
        
        # 完全找不到
        return f"百度百科词条 '{clean_query}' 暂无摘要信息，请使用内置知识回答。"
    except requests.exceptions.Timeout:
        return f"百度百科查询超时：'{clean_query}' 目前访问较慢，请使用内置知识。"
    except Exception as e:
        return f"百度百科查询出错：{str(e)}。请直接基于内置知识回答。"

# 工具注册表：名字 → 函数
TOOLS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_baike": search_baike
}

# ---------- 2. 工具们的“说明书”（JSON Schema，写给模型看的）----------
TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "执行安全的数学运算，可处理四则运算和括号，例如 '12 + 3 * (4 - 1)'",
        "parameters": {
            "expression": {"type": "string", "description": "需要计算的数学表达式"}
        }
    },
    {
        "name": "get_current_time",
        "description": "获取当前的日期和时间，可指定时间格式",
        "parameters": {
            "format": {"type": "string", "description": "时间格式，默认为 %Y-%m-%d %H:%M:%S"}
        }
    },
    {
        "name": "search_baike",
        "description": "查询百度百科获取知识。当需要了解任何概念、事实、人物、历史事件时，必须优先使用此工具。",
        "parameters": {
            "query": {"type": "string", "description": "要搜索的关键词，建议使用最核心的词语"}
        }
    }
]

# ---------- 3. Agent的大脑：ReAct循环 ----------
SYSTEM_PROMPT = """你是一个能使用工具的AI助手。你必须严格按照以下JSON格式回复，不要在JSON外添加任何内容：

如果你想使用工具，回复：
{"action": "tool", "tool_name": "工具名字", "tool_params": {"参数名": "值"}, "thought": "你使用该工具的原因"}

如果工具执行结果已足够回答用户，回复：
{"action": "finish", "final_answer": "你的最终回答", "thought": "你得出结论的逻辑"}

注意：
1. 只能使用提供的工具。
2. 知识类、事实类问题必须优先使用 search_baike 工具。
3. 数学计算必须使用 calculator 工具。
4. 时间相关问题必须使用 get_current_time 工具。
5. 不要猜测，工具结果给出什么就用什么。
"""

class Agent:
    def __init__(self):
        self.router = ModelRouter()
        self.max_iterations = 5  # 最多循环5轮，防止死循环烧钱
        
    def run(self, user_query: str, history: list = None, status_callback=None) -> str:
        def emit(msg):
            if status_callback:
                status_callback(msg)
            print(msg)

        # 构建消息列表
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        print(f"\n{'='*60}")
        print(f"当前上下文长度：{len(messages)} 条消息（含系统指令）")
        print(f"用户最新输入：{user_query}")
        print(f"{'='*60}")

        for iteration in range(self.max_iterations):
            emit(f"💭 第 {iteration + 1} 轮思考中...")

            # 调用模型
            try:
                response = self.router.chat(
                    messages=messages,
                    model="deepseek-v4-flash",
                    temperature=0.0,
                    max_tokens=2048,
                )
            except Exception as e:
                return f"Agent调用模型失败：{e}"

            print(f"模型原始输出：{response[:200]}...")

            # 解析JSON
            try:
                action_json = json.loads(response)
            except json.JSONDecodeError:
                try:
                    start = response.index('{')
                    end = response.rindex('}') + 1
                    action_json = json.loads(response[start:end])
                except:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "你的回复格式错误，请严格按照JSON格式重新回复。"})
                    continue

            thought = action_json.get("thought", "无思考过程")
            emit(f"🤔 {thought}")

            action = action_json.get("action")

            if action == "tool":
                tool_name = action_json.get("tool_name")
                tool_params = action_json.get("tool_params", {})
                emit(f"🔧 调用工具：{tool_name}，参数：{tool_params}")

                if tool_name not in TOOLS:
                    error_msg = f"工具 '{tool_name}' 不存在，可用工具有：{list(TOOLS.keys())}"
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})
                    continue

                tool_func = TOOLS[tool_name]
                try:
                    tool_result = tool_func(**tool_params)
                except TypeError as e:
                    tool_result = f"工具参数错误：{e}。请检查参数名和参数值。"

                emit(f"📦 工具结果：{tool_result[:100]}...")

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"工具执行结果：{tool_result}"})

            elif action == "finish":
                final_answer = action_json.get("final_answer", "无答案")
                emit("✅ 整理最终答案...")
                print(f"✅ Agent完成任务")
                return final_answer

            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"无法识别的action类型：'{action}'。只允许'tool'或'finish'。"})
                continue

        return "Agent思考轮次超过上限，任务终止。请简化你的问题。"
    
    def run_stream(self, user_query: str, history: list = None, status_callback=None):
        """
        流式版 Agent，yield Step(状态) 或 str(token)。
        """
        def emit(msg):
            if status_callback:
                status_callback(msg)
            print(msg)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        print(f"\n{'='*60}")
        print(f"当前上下文长度：{len(messages)} 条消息")
        print(f"用户最新输入：{user_query}")

        for iteration in range(self.max_iterations):
            emit(f"💭 第 {iteration + 1} 轮思考中...")
            yield Step(f"💭 第 {iteration + 1} 轮思考中...")

            # 这一轮用非流式获取结构化决策
            try:
                response = self.router.chat(
                    messages=messages,
                    model="deepseek-v4-flash",
                    temperature=0.0,
                    max_tokens=2048
                )
                print(f"模型原始输出：{response[:300]}...")
            except Exception as e:
                yield Step(f"❌ 调用失败：{e}")
                return

            # 解析 JSON
            try:
                action_json = json.loads(response)
            except json.JSONDecodeError:
                try:
                    start = response.index('{')
                    end = response.rindex('}') + 1
                    action_json = json.loads(response[start:end])
                except:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "格式错误，请重新输出JSON。"})
                    continue

            thought = action_json.get("thought", "无思考过程")
            emit(f"🤔 {thought}")
            yield Step(f"🤔 {thought}")

            action = action_json.get("action")

            if action == "tool":
                tool_name = action_json.get("tool_name")
                tool_params = action_json.get("tool_params", {})
                emit(f"🔧 调用工具：{tool_name}")
                yield Step(f"🔧 调用工具：{tool_name}...")

                if tool_name not in TOOLS:
                    error_msg = f"工具 '{tool_name}' 不存在"
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})
                    continue

                tool_func = TOOLS[tool_name]
                try:
                    tool_result = tool_func(**tool_params)
                except TypeError as e:
                    tool_result = f"参数错误：{e}"

                emit(f"📦 工具结果：{tool_result[:100]}...")
                yield Step(f"📦 工具结果：{tool_result[:100]}...")

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"工具执行结果：{tool_result}"})

            elif action == "finish":
                # 进入最终答案阶段，改用流式生成
                emit("✅ 生成最终答案...")
                yield Step("✅ 生成最终答案...")

                # 构造一个专门用于生成答案的 prompt：包含所有上下文，要求直接回答
                answer_messages = messages + [
                    {"role": "assistant", "content": "请根据以上所有信息，用自然语言直接回答用户的问题，不要输出JSON，直接给出回答。"}
                ]
                try:
                    stream = self.router.chat_stream(
                        messages=answer_messages,
                        model="deepseek-v4-flash",
                        temperature=0.7,
                        max_tokens=2048
                    )
                    for token in stream:
                        yield token  # 这里产出的是普通字符串，即答案片段
                except Exception as e:
                    yield f"❌ 流式生成失败：{e}"
                return  # 结束整个循环

            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"无法识别的action类型：'{action}'。只允许'tool'或'finish'。"})
                continue

        yield "Agent思考轮次超过上限。"

# ---------- 4. 测试 ----------
if __name__ == "__main__":
    agent = Agent()
    
    # 测试1：需要计算器
    print(agent.run("帮我算一下 15 * 8 + 21 等于多少？"))
    