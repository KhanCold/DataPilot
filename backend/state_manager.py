import os
import shutil
import json
from typing import List, Dict, Optional, Any

from custom_types import AllSummaries, DataFrameSummary, Plan, PlanStep, Interaction


class StateManager:
    """
    Manages the system's state, including workspace files, conversation history,
    and DataFrame summaries. This is the system's "memory".
    """
    def __init__(self, workspace_dir: str = "./workspace"):
        """
        Initializes the StateManager.

        Args:
            workspace_dir (str): The directory to use for storing and managing files.
        """
        self.workspace_dir = workspace_dir
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)
        
        # --- Core State Components ---

        # A sequential log of all user interactions.
        self.interactions: List[Interaction] = []

        # A flat list of all successfully executed code blocks from all interactions.
        self.executed_code_blocks: List[str] = []

        # DataFrame summaries, storing metadata of all DataFrames in the kernel.
        self.dataframe_summaries: AllSummaries = {}

    def _format_summary_for_llm(self, summary: DataFrameSummary) -> str:
        """Formats a single DataFrame summary into a string for the LLM context."""
        if "error" in summary:
            return f"  - Error fetching summary: {summary['error']}"

        shape = summary.get('shape', 'N/A')
        columns = summary.get('columns_and_dtypes', {})
        head_sample = summary.get('head_sample', 'N/A')

        col_str_parts = []
        for col, dtype in columns.items():
            col_str_parts.append(f"{col} ({dtype})")
        
        return (
            f"  - Shape: {shape}\n"
            f"  - Columns: [{', '.join(col_str_parts)}]\n"
            f"  - Head Sample:\n```csv\n{head_sample}```"
        )

    def start_new_interaction(self, query: str):
        """
        Starts a new interaction cycle, appending a new entry to the history.
        """
        new_interaction: Interaction = {"query": query, "plan": []}
        self.interactions.append(new_interaction)

    def set_plan(self, plan: Plan):
        """Sets the execution plan for the current interaction."""
        if not self.interactions:
            # This should not happen if start_new_interaction is called correctly
            raise ValueError("Cannot set plan without an active interaction.")
        self.interactions[-1]["plan"] = plan

    def add_executed_code_block(self, code: str, step_id: Optional[int], result: str):
        """
        Adds a successfully executed code block to the history and, if a step_id
        is provided, to the current interaction's plan step.
        """
        self.executed_code_blocks.append(code)
        
        if not self.interactions or step_id is None:
            return
            
        current_plan = self.interactions[-1]["plan"]
        for step in current_plan:
            if step["step_id"] == step_id:
                step["code"] = code
                step["result"] = result
                break

    def update_plan_step_status(self, step_id: int, status: str):
        """Updates the status of a specific plan step in the current interaction."""
        if not self.interactions:
            return

        current_plan = self.interactions[-1]["plan"]
        for step in current_plan:
            if step["step_id"] == step_id:
                step["status"] = status
                break

    def load_csvs(self, user_file_paths: List[str]):
        """
        将用户提供的 CSV 文件复制到工作区目录。

        Args:
            user_file_paths (List[str]): 用户提供的 CSV 文件路径列表。
        """
        for file_path in user_file_paths:
            if os.path.exists(file_path) and file_path.endswith('.csv'):
                try:
                    shutil.copy(file_path, self.workspace_dir)
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
            else:
                print(f"Warning: File not found or not a CSV, skipping: {file_path}")

    def update_all_dataframe_summaries(self, summaries_dict: AllSummaries):
        """
        Replaces the current DataFrame summaries entirely.
        This method is called by the Worker after the CodeExecutor runs code
        and fetches the latest state.

        Args:
            summaries_dict (AllSummaries): The new dictionary of DataFrame summaries
                                           from the kernel.
        """
        self.dataframe_summaries = summaries_dict

    def get_workspace_files(self) -> List[str]:
        """
        获取工作区目录下的所有文件名。

        Returns:
            List[str]: 文件名列表。
        """
        try:
            return [f for f in os.listdir(self.workspace_dir) if os.path.isfile(os.path.join(self.workspace_dir, f))]
        except FileNotFoundError:
            return []

    def get_planner_context(self) -> str:
        """
        Assembles a concise context string for the Planner, including history.
        """
        context_parts = []
        
        # 1. Historical Interactions (if any)
        if len(self.interactions) > 1:
            context_parts.append("**Previous Interactions History:**")
            # Iterate through all but the last (current) interaction
            for i, interaction in enumerate(self.interactions[:-1]):
                context_parts.append(f"\n--- Turn {i+1} ---")
                context_parts.append(f"User Query: {interaction['query']}")
                
                plan_summary = "\n".join(
                    [f"  - Step {s['step_id']}: {s['task']}" for s in interaction['plan']]
                )
                context_parts.append(f"Plan:\n{plan_summary}")

                executed_code = "\n".join(
                    [f"```python\n{s['code']}\n```" for s in interaction['plan'] if s.get('code')]
                )
                if executed_code:
                    context_parts.append(f"Executed Code:\n{executed_code}")
            context_parts.append("\n" + "="*20 + "\n")

        # 2. Current User Query
        if self.interactions:
            current_query = self.interactions[-1]['query']
            context_parts.append(f"**Current User Query:**\n{current_query}")
        
        # 3. Add workspace file list
        files_str = "\n".join([f"`{f}`" for f in self.get_workspace_files()])
        context_parts.append(f"\n**Workspace Files:**\n{files_str}")

        # 4. Add DataFrame summaries
        if self.dataframe_summaries:
            context_parts.append("\n**DataFrame Summaries in Sandbox:**")
            for df_name, summary in self.dataframe_summaries.items():
                formatted_summary = self._format_summary_for_llm(summary)
                context_parts.append(f"- `{df_name}`:\n{formatted_summary}\n")
        
        return "\n".join(context_parts)

    def get_worker_context(self, current_step: PlanStep) -> str:
        """
        Assembles a detailed, structured context string for the Worker.
        This provides the Worker with all the necessary information to execute a single step.
        """
        if not self.interactions:
            return "Error: No active interaction."

        context_parts = []
        current_interaction = self.interactions[-1]
        current_query = current_interaction['query']
        current_plan = current_interaction['plan']

        # 1. Overall Goal
        context_parts.append(f"**User's Current Goal:**\n{current_query}")

        # 2. Full Plan and Current Step
        plan_str = []
        for step in current_plan:
            prefix = "==>" if step['step_id'] == current_step['step_id'] else "   "
            status = step.get('status', 'pending')
            plan_str.append(f"{prefix} Step {step['step_id']} ({status}): {step['task']}")
        context_parts.append(f"\n**Full Plan (you are on Step {current_step['step_id']}):**\n" + "\n".join(plan_str))

        # 3. Previously Executed Code (from all interactions)
        if self.executed_code_blocks:
            code_history = "\n".join(
                [f"```python\n{code}\n```" for code in self.executed_code_blocks]
            )
            context_parts.append(
                "\n**Code Executed So Far (Across All Interactions):**\n"
                "The following code blocks have been successfully executed in the sandbox. "
                "Do not repeat them. You can assume their variables (e.g., DataFrames) are available.\n"
                f"{code_history}"
            )
        else:
            context_parts.append("\n**Code Executed So Far:**\nNo code has been executed yet.")

        # 4. Workspace Files
        files_str = "\n".join([f"`{f}`" for f in self.get_workspace_files()])
        context_parts.append(f"\n**Workspace Files:**\n{files_str}")

        # 5. DataFrame Summaries
        if self.dataframe_summaries:
            context_parts.append("\n**Current DataFrame Summaries in Sandbox:**")
            for df_name, summary in self.dataframe_summaries.items():
                formatted_summary = self._format_summary_for_llm(summary)
                context_parts.append(f"- `{df_name}`:\n{formatted_summary}\n")
        
        return "\n".join(context_parts)

    def get_dataframe_summaries_for_display(self) -> str:
        """
        获取一个用于向用户展示的、格式化的 DataFrame摘要字符串。
        """
        if not self.dataframe_summaries:
            return "No data loaded yet."
        
        output = ["Data loaded successfully. Here are the summaries:"]
        for name, summary in self.dataframe_summaries.items():
            output.append(f"\n--- DataFrame: {name} ---")
            # Pretty print the summary dictionary for user display
            output.append(json.dumps(summary, indent=2, ensure_ascii=False))
        return "\n".join(output)

