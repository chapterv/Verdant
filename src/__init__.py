"""
Web Agent Green Agent Package

Green Agent for evaluating Web browsing agents on AgentBeats platform
"""

__version__ = "1.0.0"
__author__ = "AgentX Team"

from .agent import (
    WebAgentGreenAgent,
    Task,
    TaskStatus,
    EvaluationResult,
    WebEnvironment,
    MetricsCalculator
)

from .a2a_protocol import (
    A2AClient,
    A2AServer,
    A2AMessage,
    A2ATask,
    A2ATaskResult,
    A2AMessageType,
    TaskState,
    create_a2a_task
)

from .webarena_benchmark import (
    WebArenaTask,
    TaskCategory,
    TaskDifficulty,
    TaskSet,
    WEBARENA_TASKS,
    get_tasks_by_category,
    get_tasks_by_difficulty,
    get_task_by_id,
    get_all_task_ids,
    to_internal_task,
    create_full_webarena_task_set,
    create_easy_task_set,
    create_ecommerce_task_set
)
from .service import create_app
from .benchmark_quality import analyze_tasks, analyze_and_save, save_report

__all__ = [
    # Main Agent
    "WebAgentGreenAgent",
    "Task",
    "TaskStatus",
    "EvaluationResult",
    "WebEnvironment",
    "MetricsCalculator",
    
    # A2A Protocol
    "A2AClient",
    "A2AServer",
    "A2AMessage",
    "A2ATask",
    "A2ATaskResult",
    "A2AMessageType",
    "TaskState",
    "create_a2a_task",
    
    # WebArena Benchmark
    "WebArenaTask",
    "TaskCategory",
    "TaskDifficulty",
    "TaskSet",
    "WEBARENA_TASKS",
    "get_tasks_by_category",
    "get_tasks_by_difficulty",
    "get_task_by_id",
    "get_all_task_ids",
    "to_internal_task",
    "create_full_webarena_task_set",
    "create_easy_task_set",
    "create_ecommerce_task_set",
    # Service
    "create_app",
    # Quality analysis
    "analyze_tasks",
    "analyze_and_save",
    "save_report",
]
