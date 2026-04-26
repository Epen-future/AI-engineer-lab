import json
from datetime import datetime
from model_router import ModelRouter
import wikipedia

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

def search_wikipedia(query: str, sentences: int = 3) -> str:
    """
    搜索维基百科，返回摘要。处理歧义和页面未找到情况。
    """
    try:
        wikipedia.set_lang("zh")
        try:
            page = wikipedia.page(query)
            summary = wikipedia.summary(query, sentences=sentences)
            return f"维基百科条目 '{page.title}'：\n{summary}\n链接：{page.url}"
        except wikipedia.DisambiguationError as e:
            options = e.options[:3]
            return f"关键词 '{query}' 有歧义，可能指的是：{', '.join(options)}。请选择更具体的词重试。"
        except wikipedia.PageError:
            search_results = wikipedia.search(query, results=3)
            if not search_results:
                return f"在维基百科中没有找到与 '{query}' 相关的结果。"
            page = wikipedia.page(search_results[0])
            summary = wikipedia.summary(search_results[0], sentences=sentences)
            return f"搜索 '{query}' 最接近的条目 '{page.title}'：\n{summary}\n链接：{page.url}"
    except Exception as e:
        return f"维基百科查询出错：{str(e)}"

def get_wikipedia_page_content(title: str, sentences: int = 10) -> str:
    """获取指定维基百科页面的详细内容。"""
    try:
        wikipedia.set_lang("zh")
        summary = wikipedia.summary(title, sentences=sentences)
        page = wikipedia.page(title)
        return f"页面 '{title}' 内容：\n{summary}\n链接：{page.url}"
    except Exception as e:
        return f"获取页面内容出错：{str(e)}"

# 工具注册表：名字 → 函数
TOOLS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_wikipedia": search_wikipedia,
    "get_wikipedia_page_content": get_wikipedia_page_content
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
        "name": "search_wikipedia",
        "description": "搜索维基百科，获取某个主题的摘要信息。当需要了解事实、概念或背景知识时使用。",
        "parameters": {
            "query": {"type": "string", "description": "搜索关键词"},
            "sentences": {"type": "integer", "description": "返回的摘要句子数，默认3"}
        }
    },
    {
        "name": "get_wikipedia_page_content",
        "description": "获取指定维基百科页面的详细内容。当search_wikipedia提示有歧义，或需要深入阅读某个具体条目时使用。",
        "parameters": {
            "title": {"type": "string", "description": "确切的维基百科页面标题"},
            "sentences": {"type": "integer", "description": "返回的摘要句子数，默认10"}
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
2. 知识类、事实类问题必须优先使用 search_wikipedia 工具。
3. 如果 search_wikipedia 返回“歧义”，应使用 get_wikipedia_page_content 明确其中一个条目。
4. 数学计算必须使用 calculator 工具。
5. 时间相关问题必须使用 get_current_time 工具。
6. 不要猜测，工具结果给出什么就用什么。
"""

class Agent:
    def __init__(self):
        self.router = ModelRouter()
        self.max_iterations = 5  # 最多循环5轮，防止死循环烧钱
        
    def run(self, user_query: str) -> str:
        """
        核心运行方法。输入用户自然语言问题，返回最终答案。
        """
        # 初始化对话历史，只有系统指令和用户第一条消息
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
        
        print(f"\n{'='*60}")
        print(f"用户：{user_query}")
        print(f"{'='*60}")
        
        for iteration in range(self.max_iterations):
            print(f"\n--- 第 {iteration + 1} 轮思考 ---")
            
            # 第1步：调用模型
            try:
                response = self.router.chat(
                    messages=messages,
                    model="deepseek-v4-flash",
                    temperature=0.0,   # 思考类任务必须降低随机性，保证格式稳定
                    max_tokens=512
                )
            except Exception as e:
                return f"Agent调用模型失败：{e}"
            
            print(f"模型原始输出：{response[:200]}...")  # 打印前200字符，方便调试
            
            # 第2步：解析模型输出的JSON
            try:
                action_json = json.loads(response)
            except json.JSONDecodeError:
                # 模型偶尔会在JSON前后加废话，尝试提取第一个{ 到最后一个}
                try:
                    start = response.index('{')
                    end = response.rindex('}') + 1
                    action_json = json.loads(response[start:end])
                except:
                    # 实在解析不了，让模型重试
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "你的回复格式错误，请严格按照JSON格式重新回复。"})
                    continue
            
            thought = action_json.get("thought", "无思考过程")
            print(f"💭 思考：{thought}")
            
            # 第3步：根据action类型分支
            action = action_json.get("action")
            
            if action == "tool":
                tool_name = action_json.get("tool_name")
                tool_params = action_json.get("tool_params", {})
                
                print(f"🔧 调用工具：{tool_name}，参数：{tool_params}")
                
                if tool_name not in TOOLS:
                    error_msg = f"工具 '{tool_name}' 不存在，可用工具有：{list(TOOLS.keys())}"
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})
                    continue
                
                # 执行工具
                tool_func = TOOLS[tool_name]
                try:
                    tool_result = tool_func(**tool_params)
                except TypeError as e:
                    tool_result = f"工具参数错误：{e}。请检查参数名和参数值。"
                
                print(f"📦 工具结果：{tool_result}")
                
                # 把模型调用和工具结果都塞回对话历史
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"工具执行结果：{tool_result}"})
                
            elif action == "finish":
                final_answer = action_json.get("final_answer", "无答案")
                print(f"✅ Agent完成任务")
                return final_answer
                
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"无法识别的action类型：'{action}'。只允许'tool'或'finish'。"})
                continue
        
        # 超过最大循环数，强制结束
        return "Agent思考轮次超过上限，任务终止。请简化你的问题。"

# ---------- 4. 测试 ----------
if __name__ == "__main__":
    agent = Agent()
    
    # 测试1：需要计算器
    print(agent.run("帮我算一下 15 * 8 + 21 等于多少？"))
    
    print("\n\n" + "="*60 + "\n")
    
    # 测试2：需要时间
    print(agent.run("现在是几点几分？"))

    # 测试3：需要查维基百科
    print(agent.run("人工智能中的Transformer是什么？"))