import json
from typing import List, Dict, Any

from llm_api import get_llm_response
from custom_types import Plan, PlanStep
# The StateManager import is no longer needed if we pass the prompt directly,
# but it's good practice if the Planner needs other things from state in the future.
# from state_manager import StateManager

class Planner:
    """
    负责任务规划和重新规划。
    它将用户的高级请求分解为一系列可执行的逻辑步骤,并在执行失败时调整计划。
    """

    def print_plan(self, plan: List[PlanStep]):
        print(f"\n[Plan]:")
        for step in plan:
            # status is optional for printing, default to pending
            status = step.get('status', 'pending')
            print(f"Step {step['step_id']} ({status}): {step['task']}")
        return

    def generate_plan(self, prompt: str) -> List[PlanStep]:
        """
        根据用户请求和当前上下文生成一个初始计划。

        Args:
            prompt (str): 由 StateManager 生成的、包含用户请求和上下文的完整 Prompt。

        Returns:
            List[Dict[str, Any]]: 一个包含计划步骤的列表。
        """
        print("Generating a new plan...")
        
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

    def replan(self, prompt: str) -> List[PlanStep]:
        """
        在任务执行失败后,生成一个全新的计划。

        Args:
            prompt (str): 由 StateManager 生成的、用于重新规划的完整 Prompt。

        Returns:
            List[Dict[str, Any]]: 一个新的计划步骤列表。
        """
        print("Previous plan failed. Generating a new plan (re-planning)...")
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