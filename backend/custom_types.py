# -*- coding: utf-8 -*-

"""
This file contains custom type definitions used across the application
to ensure type consistency and improve readability.
"""

from typing import Dict, Tuple, List, TypedDict, Union, Optional
from typing_extensions import Literal

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
    It's progressively populated as it's executed.
    """
    step_id: int
    task: str
    status: str  # "pending", "in_progress", "completed", "failed"
    code: Optional[str]
    result: Optional[str]


# A complete execution plan is a list of PlanSteps
Plan = List[PlanStep]


class Interaction(TypedDict):
    """
    Represents a single, complete interaction cycle, from user query
    to a planned and executed series of steps.
    """
    query: str
    plan: Plan


class WorkerResult(TypedDict):
    """
    Represents a successful execution result from the Worker.
    """
    status: Literal["success"]
    code: str
    result: str
    error: Optional[str]
    task: Optional[str]