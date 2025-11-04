import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import time

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

def get_llm_response(prompt: str, model: str = "qwen3-max", response_format_type: str = "json_object", timeout: int = 10) -> dict:
    """

        prompt (str): 发送给大模型的提示。
        model (str): 使用的模型名称,默认为 "qwen3-max"。
        response_format_type (str): 期望的响应格式, 'json_object' 或 'text'。
        timeout (int): 请求超时时间,单位为秒。

    Returns:
        dict or str: 从模型返回的解析后的 JSON 对象或纯文本字符串。
    """
    retries = 3
    for attempt in range(retries):
        try:
            # Call the large model, expecting a JSON formatted string in return
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": response_format_type},
                temperature=0.0,
                timeout=timeout
            )
            
            content = response.choices[0].message.content
            
            if response_format_type == "json_object":
                # Extract and parse the JSON content
                return json.loads(content)
            else:
                return content
                
        except Exception as e:
            print(f"Error getting LLM response (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(1)  # Wait for 1 second before retrying
            else:
                # If all retries fail, return a dictionary indicating failure
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
