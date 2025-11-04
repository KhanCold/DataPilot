# -*- coding: utf-8 -*-

import json
from typing import List, Dict, Any

from llm_api import get_llm_response
from custom_types import Plan, PlanStep

class Planner:
    """
    负责任务规划和重新规划。
    它将用户的高级请求分解为一系列可执行的逻辑步骤,并在执行失败时调整计划。
    """

    def _get_planner_prompt(self, user_query: str, context: str) -> str:
        """
        构建用于生成初始计划的 Planner Prompt。
        """
        return f"""
You are an expert data analysis planner. Your task is to create a concise, step-by-step plan to answer the user's request.

**IMPORTANT RULES:**
1.  **Be Concise**: Generate the minimum number of steps required. Combine related data cleaning and preparation tasks (e.g., filtering, cleaning, and type conversion) into a single, logical step.
2.  **Analyze Context**: The user's data is already available in the workspace. The context below shows available files and existing DataFrame summaries. Do NOT add a step to load data if a DataFrame with the same data already exists. The first step should be using the existing dataframes.
3.  **Final Answer**: The **last step** of the plan MUST be to summarize all findings and answer the user in natural language.

**Workspace Context:**
{context}

**User Request:**
{user_query}

**Your Plan (must be a JSON list of objects in Chinese):**
`[
    {{"step_id": 1, "task": "First logical step..."}},
    ...
    {{"step_id": N, "task": "Summarize findings and report the conclusion."}}
]`
"""

    def _get_replanner_prompt(self, user_query: str, context: str, failed_task_desc: str, error_message: str) -> str:
        """
        构建用于在任务失败后重新规划的 Re-Planner Prompt。
        """
        # Prompt is in English, as requested.
        return f"""
You are an expert data analysis re-planner. A previous plan failed to execute. Your task is to create a new, corrected, and complete plan.

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
1.  **Output Format**: You MUST return a JSON list of objects.
2.  **Correct the Error**: Your new plan must address the root cause of the error.
3.  **Completeness**: The plan should cover all steps from the current state to the final answer.
4.  **Final Answer**: The **last step** must be to summarize findings and report to the user.

**New JSON Plan (must be a JSON list of objects in Chinese):**
`[
    {{"step_id": 1, "task": "First step of the new plan..."}},
    ...
    {{"step_id": N, "task": "Summarize findings and report the conclusion."}}
]`
"""

    def print_plan(self, plan: List[PlanStep]):
        print(f"\n[Plan]:")
        for step in plan:
            # status is optional for printing, default to pending
            status = step.get('status', 'pending')
            print(f"Step {step['step_id']} ({status}): {step['task']}")
        return

    def generate_plan(self, user_query: str, context: str) -> List[PlanStep]:
        """
        根据用户请求和当前上下文生成一个初始计划。

        Args:
            user_query (str): 用户的原始数据分析请求。
            context (str): 当前的状态上下文。

        Returns:
            List[Dict[str, Any]]: 一个包含计划步骤的列表。
        """
        print("Generating a new plan...")
        prompt = self._get_planner_prompt(user_query, context)

        response_json = get_llm_response(prompt)
        
        # 健壮性检查: 确保返回的是列表
        # Robustness check: ensure the return is a list
        if isinstance(response_json, list) and all(isinstance(item, dict) for item in response_json):
            # Add default status to each step
            plan_with_status = [{**step, "status": "pending"} for step in response_json]
            self.print_plan(plan_with_status)
            return plan_with_status
        elif isinstance(response_json, dict) and 'plan' in response_json and isinstance(response_json['plan'], list):
            # 兼容 LLM 可能返回 {"plan": [...]} 的情况
            # Compatible with LLM potentially returning {"plan": [...]}
            print("Plan generated successfully (wrapped in a dict).")
            plan_with_status = [{**step, "status": "pending"} for step in response_json['plan']]
            self.print_plan(plan_with_status)
            return plan_with_status
        else:
            print(f"Error: Planner did not return a valid list of steps. Response: {response_json}")
            # 返回一个默认的失败计划
            # Return a default failure plan
            return [{"step_id": 1, "task": "Planner failed to generate a valid plan.", "status": "failed"}]

    def replan(self, user_query: str, context: str, failed_task_desc: str, error_message: str) -> List[PlanStep]:
        """
        在任务执行失败后,生成一个全新的计划。

        Args:
            user_query (str): 用户的原始请求。
            context (str): 当前的状态上下文。
            failed_task_desc (str): 失败的步骤的描述。
            error_message (str): 执行失败时返回的错误信息。

        Returns:
            List[Dict[str, Any]]: 一个新的计划步骤列表。
        """
        print("Previous plan failed. Generating a new plan (re-planning)...")
        prompt = self._get_replanner_prompt(user_query, context, failed_task_desc, error_message)
        response_json = get_llm_response(prompt)

        if isinstance(response_json, list) and all(isinstance(item, dict) for item in response_json):
            plan_with_status = [{**step, "status": "pending"} for step in response_json]
            self.print_plan(plan_with_status)
            return plan_with_status
        elif isinstance(response_json, dict) and 'plan' in response_json:
            plan_data = response_json['plan']
            if isinstance(plan_data, list):
                print("Re-plan generated successfully (wrapped in a dict).")
                plan_with_status = [{**step, "status": "pending"} for step in plan_data]
                self.print_plan(plan_with_status)
                return plan_with_status
            elif isinstance(plan_data, dict):
                # 兼容 LLM 可能返回 {"plan": {"step_1": {"description": ...}}} 的情况
                # Compatible with LLM potentially returning {"plan": {"step_1": {"description": ...}}}
                print("Re-plan generated successfully (converted from dict).")
                new_plan = []
                for key, value in sorted(plan_data.items()):
                    if isinstance(value, dict) and 'description' in value:
                        try:
                            # 从 "step_1" 中提取数字 1
                            # Extract the number 1 from "step_1"
                            step_id = int(key.split('_')[-1])
                            new_plan.append({"step_id": step_id, "task": value['description'], "status": "pending"})
                        except (ValueError, IndexError):
                            # 如果键不是预期的格式,则忽略
                            # Ignore if the key is not in the expected format
                            pass
                if new_plan:
                    return new_plan

        print(f"Error: Re-planner did not return a valid list of steps. Response: {response_json}")
        return [{"step_id": 1, "task": "Re-planner failed to generate a valid plan.", "status": "failed"}]