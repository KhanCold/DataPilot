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
        self.max_analysis_cycles = 3 # max retries for the analysis loop

    def _validate_and_summarize(self, last_code_execution_result: str) -> Dict[str, Any]:
        """
        Validates if the analysis is complete and generates a summary, or provides feedback for re-planning.
        """
        current_interaction = self.state_manager.interactions[-1]
        current_query = current_interaction['query']
        current_plan = current_interaction['plan']
        full_script = "\n".join(self.state_manager.executed_code_blocks)

        validation_prompt = self.state_manager.get_validation_prompt(
            current_query=current_query,
            current_plan=current_plan,
            full_script=full_script,
        )
        
        response = get_llm_response(validation_prompt, response_format_type='json_object', timeout=30)
        
        # Ensure the response has a status, default to incomplete if missing
        if 'status' not in response:
            response['status'] = 'incomplete'
            response['reason'] = 'The validation model returned an invalid format.'

        # Save the validation result to the state
        self.state_manager.set_validation_result(response)
            
        return response

    def load_csvs(self, user_file_paths: List[str]):
        """
        Loads CSV files into the workspace, executes code to load them into DataFrames,
        and updates the DataFrame summaries.
        """
        # 1. Copy files to workspace
        self.state_manager.load_csvs(user_file_paths)
        
        # 2. Load data into DataFrames and update summaries
        auto_generated_code_blocks = []

        # Pre-import pandas for the kernel, as it's used in almost all data analysis.
        pre_import = "import pandas as pd"
        auto_generated_code_blocks.append({"code": pre_import})

        for i, file_path in enumerate(user_file_paths):
            file_name = os.path.basename(file_path)
            print(f"Loading{file_name}...")
            if file_name.endswith('.csv'):
                df_name = f"df_{i+1}"
                code = f"{df_name} = pd.read_csv('{file_name}')"
                auto_generated_code_blocks.append({"code": code})
                
        # To ensure file I/O is in the correct path, change the working directory to the sandbox's workspace.
        workspace_path = os.path.abspath(self.state_manager.workspace_dir).replace('\\', '/')
        
        # Combine all loading code into a single block for execution
        full_code_to_run = f"import os\nos.chdir('{workspace_path}')\n"
        full_code_to_run += "\n".join([block['code'] for block in auto_generated_code_blocks])

        stdout, stderr = self.code_executor.run_code(full_code_to_run)
        
        if stderr:
            print(f"Auto-loading of CSVs failed: {stderr}")
        else:
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
        planner_prompt = self.state_manager.get_planner_prompt(user_query, planner_context)
        plan = self.planner.generate_plan(planner_prompt)
        self.state_manager.set_plan(plan)

        # Outer loop for analysis and validation cycles
        for cycle in range(self.max_analysis_cycles):
            
            # 3. Execute the plan step by step
            current_step_index = 0
            last_code_execution_result = ""
            plan_succeeded = True
            
            current_plan = self.state_manager.interactions[-1]['plan']
            while current_plan and current_plan[0].get('status') != 'failed' and current_step_index < len(current_plan):
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
                    replan_prompt = self.state_manager.get_planner_prompt(
                        user_query=self.state_manager.interactions[-1]['query'],
                        context=planner_context,
                        failed_task_desc=failed_task,
                        error_message=error_message
                    )
                    new_plan = self.planner.replan(replan_prompt)
                    # If re-planning fails, exit
                    if not new_plan or new_plan[0].get("status") == "failed":
                        print("\n[Re-planning Failed]: Could not generate a new plan. Aborting.")
                        break # Exit the inner while loop

                    self.state_manager.set_plan(new_plan)
                    current_step_index = 0  # Restart from the beginning of the new plan
                    current_plan = self.state_manager.interactions[-1]['plan'] # Refresh current_plan
            
            # If the inner loop was broken due to re-planning failure, stop the outer loop too.
            if not plan_succeeded and (not current_plan or current_plan[0].get("status") == "failed"):
                break
            
            # 4. Validate and potentially summarize the result
            validation_result = self._validate_and_summarize(last_code_execution_result)

            if validation_result.get('status') == 'complete':
                self.present_result(validation_result['summary'], last_code_execution_result)
                return # Analysis is successful and complete
            else:
                # If incomplete, use the reason to re-plan
                reason = validation_result.get('reason', 'The analysis was deemed incomplete for an unspecified reason.')
                print(f"\n[Analysis Incomplete]: {reason}. Re-planning...")
                
                planner_context = self.state_manager.get_planner_context()
                replan_prompt = self.state_manager.get_planner_prompt(
                    user_query=self.state_manager.interactions[-1]['query'],
                    context=planner_context,
                    failed_task_desc="The overall analysis did not fully answer the user's question.",
                    error_message=reason # Use the feedback as the "error" for re-planning
                )
                new_plan = self.planner.replan(replan_prompt)

                if not new_plan or new_plan[0].get("status") == "failed":
                    print("\n[Re-planning Failed]: Could not generate a new plan based on feedback. Aborting.")
                    self.present_result("The analysis could not be completed successfully.", last_code_execution_result)
                    return
                
                self.state_manager.set_plan(new_plan)
                # The loop will now continue to the next cycle with the new plan
    
        self.present_result("Failed to produce a complete analysis after multiple attempts.", last_code_execution_result)


    def present_result(self, final_answer: str, last_code_execution_result: str):
        print(f"\n[DataPilot]:\n{final_answer}")
        
        current_plan = []
        if self.state_manager.interactions:
            current_plan = self.state_manager.interactions[-1].get("plan", [])
        
        plan_summary = self.state_manager.get_plan_summary(current_plan)
        print(f"\n[Execution Result]:\n{plan_summary}")
        
        full_script = "\n".join(self.state_manager.executed_code_blocks)
        print(f"\n[Full Script]:\n{full_script}")

    def shutdown(self):
        """
        Safely shuts down the CodeExecutor's kernel.
        """
        self.code_executor.shutdown()
