# -*- coding: utf-8 -*-

from typing import List, Dict, Any
import os

from state_manager import StateManager
from code_executor import CodeExecutor
from planner import Planner
from worker import Worker
from custom_types import PlanStep

class Orchestrator:
    """
    协调所有模块,管理从用户请求到最终答案的端到端流程。
    """
    def __init__(self):
        self.state_manager = StateManager()
        self.code_executor = CodeExecutor()
        self.planner = Planner()
        self.worker = Worker(self.code_executor, self.state_manager)

    def load_csvs(self, user_file_paths: List[str]):
        """
        Loads CSV files into the workspace, executes code to load them into DataFrames,
        and updates the DataFrame summaries.
        """
        # 1. Copy files to workspace
        self.state_manager.load_csvs(user_file_paths)
        
        # 2. Load data into DataFrames and update summaries
        for file_path in user_file_paths:
            file_name = os.path.basename(file_path)
            if file_name.endswith('.csv'):
                # Create a valid DataFrame variable name from the file name.
                df_name = os.path.splitext(file_name)[0]
                df_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in df_name)
                if df_name and not df_name[0].isalpha() and df_name[0] != '_':
                    df_name = '_' + df_name
                
                code = f"import pandas as pd\n{df_name} = pd.read_csv('{file_name}')"
                
                # To ensure file I/O is in the correct path, change the working directory to the sandbox's workspace.
                workspace_path = os.path.abspath(self.state_manager.workspace_dir).replace('\\', '/')
                code_to_run = f"import os\nos.chdir('{workspace_path}')\n{code}"
                
                stdout, stderr = self.code_executor.run_code(code_to_run)
                
                if stderr:
                    error_msg = stderr
                    print(f"Auto-loading failed for {file_name}: {error_msg}")
                    # Also update conversation history so LLM knows.
                    self.state_manager.update_conversation_history(
                        "assistant",
                        f"I tried to automatically load `{file_name}` into a DataFrame named `{df_name}`, but it failed with the following error:\n```\n{error_msg}\n```"
                    )
                else:
                    print(f"Successfully loaded {file_name}.")
        
        # After attempting to load all files, update the summaries.
        df_summaries = self.code_executor.get_dataframe_summaries_from_kernel()
        self.state_manager.update_all_dataframe_summaries(df_summaries)

        # Add the auto-generated code to the executed code history
        # This provides context to the LLM that the data is already loaded.
        for file_path in user_file_paths:
            file_name = os.path.basename(file_path)
            if file_name.endswith('.csv'):
                df_name = os.path.splitext(file_name)[0]
                df_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in df_name)
                if df_name and not df_name[0].isalpha() and df_name[0] != '_':
                    df_name = '_' + df_name
                
                # Form the code and add it to the state manager's history
                code = f"import pandas as pd\n{df_name} = pd.read_csv('{file_name}')"
                self.state_manager.add_executed_code_block(code)

    def run_analysis(self, user_query: str):
        """
        运行一次完整的数据分析流程。
        user_query (str): 用户的自然语言数据分析请求。
        """
        # 1. Initialize state for the new query
        self.state_manager.set_user_query(user_query)
        
        # 2. Generate the initial plan
        planner_context = self.state_manager.get_planner_context()
        plan = self.planner.generate_plan(user_query, planner_context)
        self.state_manager.update_conversation_history("user", user_query)
        self.state_manager.set_plan(plan)

        # 3. Execute the plan step by step
        current_step_index = 0
        last_code_execution_result = ""
        while current_step_index < len(self.state_manager.plan):
            current_step = self.state_manager.plan[current_step_index]
            task_description = current_step.get("task", "No description")
            
            # Update step status to 'in_progress'
            self.state_manager.update_plan_step_status(current_step["step_id"], "in_progress")

            print(f"[Executing Step {current_step['step_id']}/{len(self.state_manager.plan)}]:\n{task_description}")

            # Get the detailed context for the worker
            worker_context = self.state_manager.get_worker_context(current_step)
            result = self.worker.execute_task(task_description, worker_context)

            if result['status'] == 'success':
                # On success, log the code, update status, and move to the next step
                self.state_manager.add_executed_code_block(result['code'])
                self.state_manager.update_plan_step_status(current_step["step_id"], "completed")
                if result.get('result'): # If there is any stdout, store it
                    last_code_execution_result = result['result']
                self.state_manager.update_conversation_history(
                    "assistant", 
                    f"Step {current_step['step_id']} completed. Result:\n{result['result']}"
                )
                current_step_index += 1

            elif result['status'] == 'final_answer':
                # If the worker provides the final answer, print it and the full script, then exit.
                final_answer = result['result']
                print(f"[Data Copilot]:{final_answer}")
                
                # Also print the last code execution result
                print("<Code Execution Result>:")
                print(last_code_execution_result)
                print("</Code Execution Result>")

                # Also print the full, combined code script
                print("<Full Executed Code Script>:")
                full_script = "\n".join(self.state_manager.executed_code_blocks)
                print(full_script)
                print("</Full Executed Code Script>")

                self.state_manager.update_conversation_history("assistant", final_answer)
                break  # Exit the loop on success

            elif result['status'] == 'failed':
                # On failure, trigger the re-planning process
                error_message = result['error']
                failed_task = result['task']
                print(f"[Step {current_step['step_id']} Failed]: {error_message}")
                self.state_manager.update_plan_step_status(current_step["step_id"], "failed")
                self.state_manager.update_conversation_history(
                    "assistant",
                    f"Step {current_step['step_id']} failed with error: {error_message}. I will now try to re-plan."
                )

                # Get planner context and re-plan
                planner_context = self.state_manager.get_planner_context()
                new_plan = self.planner.replan(
                    user_query=self.state_manager.user_query,
                    context=planner_context,
                    failed_task_desc=failed_task,
                    error_message=error_message
                )
                self.state_manager.set_plan(new_plan)
                current_step_index = 0  # Restart from the beginning of the new plan
                print("Re-planning complete. Starting new plan.")


    def shutdown(self):
        """
        安全地关闭 CodeExecutor 的内核。
        """
        self.code_executor.shutdown()
