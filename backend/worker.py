import json
import os
from typing import Dict, Any

from code_executor import CodeExecutor
from state_manager import StateManager
from llm_api import get_llm_response
from custom_types import WorkerResult

class Worker:
    """
    执行 Planner 分配的单个逻辑步骤。
    内置"微型纠错"循环,并通过结构化的方式调用工具。
    """

    def __init__(self, code_executor: CodeExecutor, state_manager: StateManager):
        """
        初始化 Worker。

        Args:
            code_executor (CodeExecutor): 用于执行代码的代码执行器。
            state_manager (StateManager): 用于管理系统状态的状态管理器。
        """
        self.code_executor = code_executor
        self.state_manager = state_manager
        
        # Tool registry, mapping tool names to their corresponding handler methods
        self.tool_registry = {
            "execute_python": self._tool_execute_python,
        }

    def _tool_execute_python(self, code: str) -> Dict[str, Any]:
        """
        执行 Python 代码的工具,并自动同步执行后的状态。
        这是一个关键的封装,确保了状态的一致性。
        """
        print(f"\n[Executing Code]:\n{code}")
        # Change working directory to the sandbox's workspace to ensure file I/O is at the correct path
        workspace_path = os.path.abspath(self.state_manager.workspace_dir).replace('\\', '/')
        code_to_run = f"import os\nos.chdir('{workspace_path}')\n{code}"
        
        stdout, stderr = self.code_executor.run_code(code_to_run)
        
        # Automatically sync state
        df_summaries = self.code_executor.get_dataframe_summaries_from_kernel()
        self.state_manager.update_all_dataframe_summaries(df_summaries)

        if stderr:
            print(f"\n[Execution Error]:\n{stderr}")
            return {"status": "error", "error": stderr, "code": code}
        else:
            print(f"\n[Execution Output]:\n{stdout}")
            return {"status": "success", "result": stdout, "code": code}
        
    def execute_task(self, task_description: str, context: str, max_retries: int = 3) -> WorkerResult:
        """
        执行单个任务,包含一个 ReAct 风格的重试循环以进行"微观纠错"。

        Args:
            task_description (str): Planner 分配的具体任务描述。
            context (str): 当前的状态上下文。
            max_retries (int): 在将问题升级到 Planner 之前,Worker 的最大重试次数。

        Returns:
            WorkerResult: 一个包含任务执行结果的字典,包括状态('success' 或 'failed')。
        """
        retry_count = 0
        current_context = context
        
        while retry_count < max_retries:
            prompt = self.state_manager.get_worker_prompt(task_description, current_context)

            # print(f"\n===== Prompt:  =====")
            # print(prompt)
            # print("======= Prompt End =======")
            llm_response = get_llm_response(prompt)
            
            if "error" in llm_response or "tool_call" not in llm_response:
                error_message = llm_response.get("error", "Invalid JSON response from LLM.")
                print(f"Worker LLM call failed: {error_message}")
                retry_count += 1
                current_context += f"\n\n**Attempt {retry_count} Error:**\nFailed to get a valid tool call from LLM. Error: {error_message}"
                continue

            tool_call = llm_response["tool_call"]
            tool_name = tool_call.get("tool_name")
            arguments = tool_call.get("arguments", {})
            
            # Keep track of the code that is about to be executed
            last_executed_code = arguments.get("code", "")

            if tool_name not in self.tool_registry:
                error_message = f"Tool '{tool_name}' not found."
                print(error_message)
                retry_count += 1
                current_context += f"\n\n**Attempt {retry_count} Error:**\n{error_message}"
                continue
                
            try:
                # Call the tool
                tool_function = self.tool_registry[tool_name]
                result = tool_function(**arguments)
                
                if result.get("status") == "error":
                    # Code execution error, enter retry loop
                    retry_count += 1
                    error_feedback = result.get("error", "An unknown error occurred.")
                    last_executed_code = result.get("code", "N/A")

                    # Construct a detailed error message for the next prompt
                    error_context = (
                        f"\n\n**Attempt {retry_count} Failed:**\n"
                        f"I tried to execute the following code:\n"
                        f"```python\n{last_executed_code}\n```\n"
                        f"However, it failed with the following error:\n"
                        f"```\n{error_feedback}\n```\n"
                        f"Please analyze the error, review the current DataFrame summaries, and provide new, corrected code that is idempotent (i.e., it can be run multiple times without causing new errors)."
                    )
                    current_context += error_context
                    print(f"Task execution failed. Retrying ({retry_count}/{max_retries})...")
                    continue
                else:
                    # Tool executed successfully or returned the final answer
                    return result

            except TypeError as e:
                # Errors like argument mismatch
                error_message = f"Tool call argument error: {e}"
                print(error_message)
                retry_count += 1
                current_context += f"\n\n**Attempt {retry_count} Error:**\n{error_message}"
                continue
        
        # If all retries fail, report failure to the Orchestrator
        return {"status": "failed", "error": "Worker failed to execute the task after multiple retries.", "task": task_description}
