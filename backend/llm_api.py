import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# Load environment variables from .env file
load_dotenv()

# 从环境变量中获取 DashScope API 密钥
# Get DashScope API key from environment variables
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY not found in .env file. Please make sure to set it.")

# 初始化 OpenAI 客户端,指向 DashScope 的服务
# Initialize OpenAI client, pointing to DashScope's service
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def get_llm_response(prompt: str, model: str = "qwen3-max") -> dict:
    """
    调用大模型并获取结构化的 JSON 响应。

    Args:
        prompt (str): 发送给大模型的提示。
        model (str): 使用的模型名称,默认为 "qwen3-max"。

    Returns:
        dict: 从模型返回的解析后的 JSON 对象。
    """
    # 调用大模型,期望返回的是 JSON 格式的字符串
    # Call the large model, expecting a JSON formatted string in return
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        # 提取并解析 JSON 内容
        # Extract and parse the JSON content
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        # 如果 LLM 调用或 JSON 解析失败,打印错误并返回一个表示失败的字典
        # If the LLM call or JSON parsing fails, print the error and return a dictionary indicating failure
        print(f"Error getting LLM response or parsing JSON: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # 测试代码
    test_prompt = """
    You are a helpful assistant. Return a JSON object with a key "greeting" and value "hello".
    
    Respond with JSON only.
    
    {
        "greeting": "hello"
    }
    """
    response_json = get_llm_response(test_prompt)
    print("LLM Response:")
    print(response_json)
