# -*- coding: utf-8 -*-

import json
import time
from jupyter_client.manager import KernelManager

from custom_types import AllSummaries

class CodeExecutor:
    """
    管理一个隔离且有状态的 Python 执行环境。
    通过 jupyter_client 与一个 ipykernel 内核进行通信,以实现代码的执行和状态的维护。
    """

    def __init__(self):
        """
        初始化 CodeExecutor。
        启动一个 KernelManager 和一个 KernelClient,并建立与内核的通信通道。
        """
        # 启动一个内核管理器,它会自动寻找可用的 ipykernel
        self.km = KernelManager()
        self.km.start_kernel()
        
        # 获取内核客户端,用于与内核通信
        self.kc = self.km.client()
        self.kc.start_channels()
        
        # 等待内核确认已准备就绪
        try:
            self.kc.wait_for_ready(timeout=60)
            print("Kernel is ready.")
        except RuntimeError:
            print("Timeout waiting for kernel to be ready.")
            self.shutdown()
            raise

    def run_code(self, code_string: str) -> tuple[str, str]:
        """
        在内核中执行一段 Python 代码。

        Args:
            code_string (str): 要执行的 Python 代码字符串。

        Returns:
            tuple[str, str]: 一个包含 stdout 和 stderr 的元组。
        """
        # 向内核发送执行请求
        msg_id = self.kc.execute(code_string)
        
        stdout_str = ""
        stderr_str = ""
        
        # 循环监听 iopub 通道以获取执行结果
        while True:
            try:
                # 从 iopub 通道获取消息
                msg = self.kc.get_iopub_msg(timeout=60)
                
                # 检查消息是否与我们的请求相关
                if msg['parent_header'].get('msg_id') == msg_id:
                    msg_type = msg['header']['msg_type']
                    content = msg['content']
                    
                    if msg_type == 'stream':
                        # 收集 stdout 和 stderr
                        if content['name'] == 'stdout':
                            stdout_str += content['text']
                        else:
                            stderr_str += content['text']
                    elif msg_type == 'display_data':
                        # 收集 display_data,例如 matplotlib 的图表
                        if 'text/plain' in content['data']:
                            stdout_str += content['data']['text/plain']
                    elif msg_type == 'execute_result':
                         if 'text/plain' in content['data']:
                            stdout_str += content['data']['text/plain']
                    elif msg_type == 'error':
                        # 收集错误信息
                        stderr_str += f"{content['ename']}: {content['evalue']}\n"
                        stderr_str += "\n".join(content['traceback'])
                    elif msg_type == 'status' and content['execution_state'] == 'idle':
                        # 当内核变为空闲状态时,表示执行完成
                        break
            except Exception:
                # 如果超时未收到消息,则认为执行结束
                break
        
        return stdout_str, stderr_str

    def get_dataframe_summaries_from_kernel(self) -> AllSummaries:
        """
        在内核中执行一个内省脚本,以获取所有 pandas DataFrame 变量的摘要。

        Returns:
            dict: 一个包含 DataFrame 摘要的字典。如果出错则为空字典。
        """
        # 内省脚本: 查找所有 DataFrame 并生成摘要
        # Introspection script: find all DataFrames and generate summaries
        introspection_script = """
import json

# __STATE_UPDATE__ 是一个魔法字符串,帮助我们从 stdout 中定位 JSON 输出
# __STATE_UPDATE__ is a magic string to help us locate the JSON output from stdout
def get_all_df_summaries():
    summaries = {}
    # 查找全局命名空间中的所有 pandas DataFrame 对象
    # Find all pandas DataFrame objects in the global namespace
    for name, var in globals().items():
        if isinstance(var, pd.DataFrame) and not name.startswith('_'):
            try:
                shape = var.shape
                # 获取列名及其数据类型
                # Get column names and their data types
                columns_and_dtypes = {col: str(var[col].dtype) for col in var.columns}
                # 获取 DataFrame 头部的少量样本作为字符串
                # Get a small sample of the DataFrame's head as a string
                head_sample = var.head(3).to_csv(index=False)
                
                summaries[name] = {
                    'shape': shape,
                    'columns_and_dtypes': columns_and_dtypes,
                    'head_sample': head_sample
                }
            except Exception as e:
                # 如果摘要生成失败,记录错误
                # If summary generation fails, record the error
                summaries[name] = {'error': str(e)}
    
    # 将摘要以带有魔法前缀的 JSON 字符串形式打印出来
    # Print the summaries as a JSON string with a magic prefix
    print(f"__STATE_UPDATE__:{json.dumps(summaries)}")

get_all_df_summaries()
        """
        
        # 运行内省脚本并捕获输出
        stdout, stderr = self.run_code(introspection_script)
        
        if stderr:
            print(f"Error during dataframe introspection: {stderr}")
            return {}

        # 从 stdout 中解析出 JSON 摘要
        for line in stdout.splitlines():
            if line.startswith("__STATE_UPDATE__:"):
                json_str = line.replace("__STATE_UPDATE__:", "")
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode dataframe summaries JSON: {e}")
                    return {}
        return {}

    def shutdown(self):
        """
        关闭内核客户端和管理器,释放资源。
        """
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)
        print("Kernel shutdown complete.")