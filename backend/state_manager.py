import os
import shutil
import json
from typing import List, Dict, Optional, Any

from custom_types import AllSummaries, DataFrameSummary, Plan, PlanStep, Interaction
from debug_utils import log_prompt_to_file


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

    # ==============================================================================
    # Private Helper Methods for Context Formatting
    # ==============================================================================

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

    def _get_formatted_workspace_files(self) -> str:
        """Returns a formatted string of workspace files for the LLM context."""
        files = self.get_workspace_files()
        if not files:
            return "**Workspace Files:**\nNo files in workspace."
        files_str = "\n".join([f"- `{f}`" for f in files])
        return f"**Workspace Files:**\n{files_str}"

    def _get_formatted_dataframe_summaries(self) -> str:
        """Returns a formatted string of all DataFrame summaries for the LLM context."""
        if not self.dataframe_summaries:
            return ""
        
        context_parts = ["\n**DataFrame Summaries in Sandbox:**"]
        for df_name, summary in self.dataframe_summaries.items():
            formatted_summary = self._format_summary_for_llm(summary)
            context_parts.append(f"- `{df_name}`:\n{formatted_summary}\n")
        return "\n".join(context_parts)

    def _get_formatted_code_history(self) -> str:
        """Returns a formatted string of all previously executed code."""
        if not self.executed_code_blocks:
            return "\n**Code Executed So Far:**\nNo code has been executed yet."

        code_history = "\n".join(
            [f"```python\n{code}\n```" for code in self.executed_code_blocks]
        )
        return (
            "\n**Code Executed So Far (Across All Interactions):**\n"
            "The following code blocks have been successfully executed in the sandbox. "
            "Do not repeat them. You can assume their variables (e.g., DataFrames) are available.\n"
            f"{code_history}"
        )

    # ==============================================================================
    # Public State Management Methods
    # ==============================================================================

    def start_new_interaction(self, query: str):
        """
        Starts a new interaction cycle, appending a new entry to the history.
        """
        new_interaction: Interaction = {"query": query, "plan": [], "validation_result": None}
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

    def set_validation_result(self, result: Dict[str, str]):
        """
        设置当前交互的最终验证结果。
        这个结果来自于Orchestrator的`_validate_and_summarize`方法。

        Args:
            result (Dict[str, str]): 包含验证状态和`summary`或`reason`的字典。
        """
        if not self.interactions:
            raise ValueError("Cannot set validation result without an active interaction.")
        self.interactions[-1]["validation_result"] = result

    def _get_formatted_history(self) -> str:
        """
        Formats the interaction history into a string for prompt context.
        It includes the query, plan, and final validation result for each past interaction.
        """
        # We iterate through all but the most recent interaction, as that's the current one.
        # The history should only contain past, completed interactions.
        if len(self.interactions) <= 1:
            return ""

        history_parts = ["## Past Interaction History:"]
        for i, interaction in enumerate(self.interactions[:-1]):
            history_parts.append(f"### Interaction {i+1}:")
            history_parts.append(f"- **User Query:** {interaction['query']}")
            
            # Format the plan
            history_parts.append("- **Plan:**")
            if interaction.get('plan'):
                for step in interaction['plan']:
                    status = step.get('status', 'pending')
                    history_parts.append(f"  - Step {step['step_id']} ({status}): {step['task']}")
            else:
                history_parts.append("  - No plan was generated.")
            
            # Format the validation result
            validation_result = interaction.get('validation_result')
            if validation_result:
                result_status = validation_result.get('status', 'N/A')
                result_summary = validation_result.get('summary', validation_result.get('reason', 'No details provided.'))
                history_parts.append(f"- **Final Result:** {result_status.capitalize()}")
                history_parts.append(f"- **Summary/Reason:** {result_summary}")
            else:
                history_parts.append("- **Final Result:** Not available.")
        
        return "\n".join(history_parts)

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
        Assembles a concise context string for the Planner.
        This focuses on the *current state* of the workspace (files and dataframes).
        Historical context is provided separately by `_get_formatted_history`.
        """
        context_parts = [
            self._get_formatted_workspace_files(),
            self._get_formatted_dataframe_summaries()
        ]
        return "\n".join(filter(None, context_parts))

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
        context_parts = [f"**User's Current Goal:**\n{current_query}"]

        # 2. Full Plan and Current Step
        plan_str_parts = []
        for step in current_plan:
            prefix = "==>" if step['step_id'] == current_step['step_id'] else "   "
            status = step.get('status', 'pending')
            plan_str_parts.append(f"{prefix} Step {step['step_id']} ({status}): {step['task']}")
        
        context_parts.append(f"\n**Full Plan (you are on Step {current_step['step_id']}):**\n" + "\n".join(plan_str_parts))

        # 3. Previously Executed Code
        context_parts.append(self._get_formatted_code_history())

        # 4. Workspace Files
        context_parts.append("\n" + self._get_formatted_workspace_files())

        # 5. DataFrame Summaries
        context_parts.append(self._get_formatted_dataframe_summaries())
        
        return "\n".join(filter(None, context_parts))

    # ==============================================================================
    # Prompt Generation Section
    # ==============================================================================

    def get_planner_prompt(self, user_query: str, context: str, failed_task_desc: Optional[str] = None, error_message: Optional[str] = None) -> str:
        """
        构建用于生成或修正计划的Prompt。
        这是一个合并后的Prompt,既能处理初次规划,也能在之前计划失败时进行重新规划。

        Args:
            user_query (str): 用户的原始请求。
            context (str): `get_planner_context` 生成的当前工作区和历史记录的上下文。
            failed_task_desc (Optional[str]): 如果是重新规划,这里会提供失败任务的描述。
            error_message (Optional[str]): 如果是重新规划,这里会提供失败任务的具体错误信息。

        Returns:
            str: 一个完整的、可以直接发送给LLM的Prompt字符串。
        """

        history = self._get_formatted_history()
        prompt = ""
        if failed_task_desc and error_message:
            # Re-planning scenario
            prompt = f"""
You are an expert data analysis re-planner. A previous plan failed to execute. Your task is to create a new, corrected, and complete plan.

{history}

**Original User Request:**
{user_query}

**Context of the Failure:**
- **Failed Step:** {failed_task_desc}
- **Error Message:** {error_message}

**Current Workspace Context:**
{context}

**Your Task:**
Based on the failure, create a **new and complete** plan to fulfill the user's request. The plan must be concise and correct the error.

**IMPORTANT RULES:**
1.  **Be Concise**: Generate the minimum number of steps required.
2.  **Output Format**: You MUST return a JSON list of objects.
3.  **Correct the Error**: Your new plan must address the root cause of the error.
4.  **Completeness**: The plan should cover all steps from the current state to the final answer.
5.  **Focus on Code**: The plan should only contain steps that can be executed as code. Do not include any plotting steps.

**New JSON Plan (must be a JSON list of objects in Chinese):**
`[
    {{"step_id": 1, "task": "First step of the new plan..."}},
    ...
    {{"step_id": N, "task": "Last step of the new plan..."}}
]`
"""
        else:
            # Initial planning scenario
            prompt = f"""
You are an expert data analysis planner. Your task is to create a concise, step-by-step plan to answer the user's request.

{history}

**IMPORTANT RULES:**
1.  **Be Concise**: Generate the minimum number of steps required. If the entire request can be accomplished in a single step, create only one step. Combine related tasks into a single, logical step.
2.  **Analyze Context**: The user's data is already available in the workspace. The context below shows available files and existing DataFrame summaries. Do NOT add a step to load data if a DataFrame with the same data already exists. The first step should be using the existing dataframes.
3.  **Focus on Code**: The plan should only contain steps that can be executed as code. Do not include any plotting steps.

**Workspace Context:**
{context}

**User Request:**
{user_query}

**Your Plan (must be a JSON list of objects in Chinese):**
`[
    {{"step_id": 1, "task": "First logical step..."}},
    ...
    {{"step_id": N, "task": "Last logical step..."}}
]`
"""
        log_prompt_to_file("planner_prompt", prompt)
        return prompt

    def get_worker_prompt(self, task_description: str, context: str) -> str:
        """
        构建用于Worker执行单个任务的Prompt。
        这个Prompt指示Worker调用`execute_python`工具来完成具体的编码任务。

        Args:
            task_description (str): 当前需要执行的任务的详细描述,来自计划(Plan)中的一步。
            context (str): `get_worker_context` 生成的、为当前任务定制的上下文信息。

        Returns:
            str: 一个完整的、可以直接发送给LLM的Prompt字符串。
        """
        prompt = f"""
You are an expert Python data analysis executor. Your task is to execute a single step in a larger plan.

Your **only** goal is to complete the current step: **{task_description}**

Carefully review the context provided below, especially the code that has already been executed.
Then, write the Python code needed to complete the current step.

You must call a tool to complete the task. Your response must be a single JSON object with 'thought' and 'tool_call'.

**## Available Tools**

1. `execute_python(code: str)`
    - Description: Executes Python code in a stateful sandbox. The sandbox has `pandas` installed.
    - Use `print()` to output text results.
    - The sandbox remembers variables from previous executions (e.g., `df`).

**## Workspace Context**
{context}

**## Instructions for Your Response**
1.  **Focus**: Write code ONLY for the current task: `{task_description}`.
2.  **Idempotency**: Ensure your code is idempotent. It should be safely runnable multiple times without causing errors.
3.  **No Repetition**: DO NOT repeat code that has already been executed, including library imports like `import pandas as pd`. You can use all variables and DataFrames created in previous steps.
4.  **MANDATORY**: The code you generate MUST end with a `print()` statement to output the final result. Do not omit this step.

**## Your Response (JSON):**
```json
{{
  "thought": "I will analyze the task and decide which tool to use...",
  "tool_call": {{
    "tool_name": "execute_python",
    "arguments": {{
      "arg1_name": "value1"
    }}
  }}
}}
```
"""
        log_prompt_to_file("worker_prompt", prompt)
        return prompt

    def get_validation_prompt(self, current_query: str, current_plan: Plan, full_script: str) -> str:
        """
        构建用于Orchestrator验证最终结果的Prompt。
        这个Prompt要求LLM判断整个分析流程是否完整地回答了用户的问题,并给出总结或指出不足。

        Args:
            current_query (str): 用户当前轮的原始请求。
            current_plan (Plan): 当前执行的完整计划,包含每一步的结果。
            full_script (str): 本轮交互中所有被执行过的Python代码的完整脚本。

        Returns:
            str: 一个完整的、可以直接发送给LLM的Prompt字符串。
        """

        history = self._get_formatted_history()

        plan_summary_parts = []
        for s in current_plan:
            plan_summary_parts.append(f"  - Step {s['step_id']} ({s.get('status', 'N/A')}): {s['task']}")
            if s.get('result'):
                plan_summary_parts.append(f"    - Result:\n```\n{s['result']}\n```\n")
        plan_summary = "\n".join(plan_summary_parts)


        prompt = f"""
You are an expert data analyst. Your task is to assess whether a data analysis task has been successfully completed and, if so, provide a summary.

{history}

**User's Original Question:**
---
{current_query}
---

**Analysis Plan and Results:**
---
{plan_summary}
---

**Executed Python Script:**
---
<script>
{full_script}
</script>
---

**Assessment Task:**

1.  **Analyze the results:** Carefully review the user's question, the analysis plan with its step-by-step results, and the full executed script.
2.  **Determine Completeness:** Has the user's question been fully and comprehensively answered, in line with the plan and its results?
3.  **Provide a JSON Response:** Based on your assessment, respond in one of the following JSON formats:

    *   **If the question is fully answered, provide a summary:**
        ```json
        {{
            "status": "complete",
            "summary": "Generate a detailed analysis report summary in Chinese. The report should be well-structured, easy to understand, and present the analysis results directly, without any meta-commentary on the analysis process itself (e.g., do not say 'This analysis successfully answered...'). The report should include the following sections:\n\n1.  **Analysis Overview**: Briefly describe the key analytical steps taken to answer the user's question.\n2.  **Key Findings**: Clearly list the main insights and results derived from the data. Use bullet points where appropriate.\n3.  **Conclusion**: Based on the findings, provide a direct and clear answer to the user's original question."
        }}
        ```

    *   **If the question is NOT fully answered, explain what's missing:**
        ```json
        {{
            "status": "incomplete",
            "reason": "Explain in Chinese why the analysis is incomplete. For example, 'The analysis only shows the total sales, but it doesn't break it down by product category as requested.' or 'The code produced an empty chart, which doesn't answer the user's question.'"
        }}
        ```
"""
        log_prompt_to_file("validation_prompt", prompt)
        return prompt

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

