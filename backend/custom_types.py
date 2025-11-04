# -*- coding: utf-8 -*-

"""
This file contains custom type definitions used across the application
to ensure type consistency and improve readability.
"""

from typing import Dict, Tuple, List, TypedDict, Union

class SuccessSummary(TypedDict):
    """
    Represents a successful summary of a DataFrame, including its shape,
    column types, and a sample of its head.
    """
    shape: Tuple[int, int]
    columns_and_dtypes: Dict[str, str]
    head_sample: str

class ErrorSummary(TypedDict):
    """
    Represents a failed attempt to summarize a DataFrame,
    containing only an error message.
    """
    error: str

# A summary for a single DataFrame can be either a success or an error
DataFrameSummary = Union[SuccessSummary, ErrorSummary]

# A dictionary containing summaries for all DataFrames in the kernel
AllSummaries = Dict[str, DataFrameSummary]


class PlanStep(TypedDict):
    """
    Represents a single step in an execution plan,
    containing a step ID and a task description.
    """
    step_id: int
    task: str
    status: str # "pending", "in_progress", "completed", "failed"

# A complete execution plan is a list of PlanSteps
Plan = List[PlanStep]
