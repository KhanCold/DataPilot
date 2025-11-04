from typing import List, Dict, Any
import os

from state_manager import StateManager
from code_executor import CodeExecutor
from planner import Planner
from worker import Worker
from custom_types import PlanStep
from llm_api import get_llm_response

class Orchestrator:
    """
    协调所有模块,管理从用户请求到最终答案的端到端流程。
    """
    def __init__(self):
        self.state_manager = StateManager()
        self.code_executor = CodeExecutor()
        self.planner = Planner()
        self.worker = Worker(self.code_executor, self.state_manager)

    def _generate_final_summary(self, last_code_execution_result: str) -> str:
        """
        Generates the final natural language summary after all steps succeed.
        """
        
        current_query = self.state_manager.interactions[-1]['query']
        full_script = "\n".join(self.state_manager.executed_code_blocks)

        summary_prompt = f"""
You are an expert data analyst. Your task is to provide a comprehensive, easy-to-understand summary in Chinese for a data analysis request.
A user asked the following question:
---
{current_query}
---

To answer this question, the following Python script was executed:
---
<script>
{full_script}
</script>
---

The script produced the following final output:
---
<output>
{last_code_execution_result}
</output>
---

Based on all the information above, please provide a final, natural-language answer to the user's original question.
- Explain what was done.
- Present the key findings.
- Directly answer the user's question.
- Your entire response must be in Chinese.
"""
        summary = get_llm_response(summary_prompt, response_format_type='text')
        return summary

    def load_csvs(self, user_file_paths: List[str]):
        """
        Loads CSV files into the workspace, executes code to load them into DataFrames,
        and updates the DataFrame summaries.
        """
        # 1. Copy files to workspace
        self.state_manager.load_csvs(user_file_paths)
        
        # 2. Load data into DataFrames and update summaries
        auto_generated_code_blocks = []
        for file_path in user_file_paths:
            file_name = os.path.basename(file_path)
            if file_name.endswith('.csv'):
                # Create a valid DataFrame variable name from the file name.
                df_name = os.path.splitext(file_name)[0]
                df_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in df_name)
                if df_name and not df_name[0].isalpha() and df_name[0] != '_':
                    df_name = '_' + df_name
                
                code = f"import pandas as pd\n{df_name} = pd.read_csv('{file_name}')"
                auto_generated_code_blocks.append({"code": code, "file_name": file_name})
                
        # To ensure file I/O is in the correct path, change the working directory to the sandbox's workspace.
        workspace_path = os.path.abspath(self.state_manager.workspace_dir).replace('\\', '/')
        
        # Combine all loading code into a single block for execution
        full_code_to_run = f"import os\nos.chdir('{workspace_path}')\n"
        full_code_to_run += "\n".join([block['code'] for block in auto_generated_code_blocks])

        stdout, stderr = self.code_executor.run_code(full_code_to_run)
        
        if stderr:
            print(f"Auto-loading of CSVs failed: {stderr}")
        else:
            print("Successfully loaded all CSV files.")
            # Add the auto-generated code to the executed code history
            # This provides context to the LLM that the data is already loaded.
            for block in auto_generated_code_blocks:
                self.state_manager.add_executed_code_block(
                    code=block['code'], 
                    step_id=None, 
                    result=stdout if not stderr else stderr
                )

        # After attempting to load all files, update the summaries.
        df_summaries = self.code_executor.get_dataframe_summaries_from_kernel()
        self.state_manager.update_all_dataframe_summaries(df_summaries)

    def run_analysis(self, user_query: str):
        """
        Runs a full data analysis flow for a given user query.
        user_query (str): The user's natural language data analysis request.
        """
        # 1. Initialize state for the new query
        self.state_manager.start_new_interaction(user_query)
        
        # 2. Generate the initial plan
        planner_context = self.state_manager.get_planner_context()
        plan = self.planner.generate_plan(user_query, planner_context)
        self.state_manager.set_plan(plan)

        # 3. Execute the plan step by step
        current_step_index = 0
        last_code_execution_result = ""
        plan_succeeded = True

        current_plan = self.state_manager.interactions[-1]['plan']
        while current_step_index < len(current_plan):
            current_step = current_plan[current_step_index]
            task_description = current_step.get("task", "No description")
            
            # Update step status to 'in_progress'
            self.state_manager.update_plan_step_status(current_step["step_id"], "in_progress")

            print(f"\n[Executing Step {current_step['step_id']}/{len(current_plan)}]:\n{task_description}")

            # Get the detailed context for the worker
            worker_context = self.state_manager.get_worker_context(current_step)
            result = self.worker.execute_task(task_description, worker_context)

            if result['status'] == 'success':
                # On success, log the code, update status, and move to the next step
                self.state_manager.add_executed_code_block(
                    code=result['code'],
                    step_id=current_step["step_id"],
                    result=result.get('result', '')
                )
                self.state_manager.update_plan_step_status(current_step["step_id"], "completed")
                if result.get('result'): # If there is any stdout, store it
                    last_code_execution_result = result['result']
                current_step_index += 1

            elif result['status'] == 'failed':
                # On failure, trigger the re-planning process
                plan_succeeded = False
                error_message = result['error']
                failed_task = result['task']
                print(f"\n[Step {current_step['step_id']} Failed]:\n{error_message}")
                self.state_manager.update_plan_step_status(current_step["step_id"], "failed")
                
                # Get planner context and re-plan
                planner_context = self.state_manager.get_planner_context()
                new_plan = self.planner.replan(
                    user_query=self.state_manager.interactions[-1]['query'],
                    context=planner_context,
                    failed_task_desc=failed_task,
                    error_message=error_message
                )
                # If re-planning fails, exit
                if not new_plan or new_plan[0].get("status") == "failed":
                    print("\n[Re-planning Failed]: Could not generate a new plan. Aborting.")
                    plan_succeeded = False
                    break

                self.state_manager.set_plan(new_plan)
                current_step_index = 0  # Restart from the beginning of the new plan
                current_plan = self.state_manager.interactions[-1]['plan'] # Refresh current_plan
        
        # 4. Generate final summary if the plan executed successfully
        if plan_succeeded:
            final_answer = self._generate_final_summary(last_code_execution_result)
            
            print("\n[Data Copilot]:\n{final_answer}")
            
            # Also print the last code execution result
            print("\n[最后执行结果]:")
            print(last_code_execution_result)

            # Also print the full, combined code script
            print("\n[完整执行代码]:")
            full_script = "\n".join(self.state_manager.executed_code_blocks)
            print(full_script)
            print("\n" + "="*30)

    def shutdown(self):
        """
        Safely shuts down the CodeExecutor's kernel.
        """
        self.code_executor.shutdown()
