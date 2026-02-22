"""Production-ready Green Agent evaluation engine."""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .a2a_protocol import A2AClient, A2ATaskResult, TaskState, create_a2a_task

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status in the Green Agent."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a single benchmark task."""

    id: str
    description: str
    target_url: str
    expected_result: str
    timeout: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Task":
        """Build a task from dictionary payload."""
        return cls(
            id=payload["id"],
            description=payload.get("description", ""),
            target_url=payload.get("target_url", ""),
            expected_result=payload.get("expected_result", ""),
            timeout=int(payload.get("timeout", 300)),
            metadata=payload.get("metadata", {}) or {},
        )


@dataclass
class EvaluationResult:
    """Evaluation output for one task and one run."""

    task_id: str
    run_id: int
    status: TaskStatus
    success: bool
    execution_time: float
    metrics: Dict[str, float]
    attempts: int = 1
    error_message: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)
    raw_output: Optional[Any] = None
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dict."""
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class WebEnvironment:
    """
    Lightweight evaluation environment.

    This environment is deterministic and controller-friendly. It does not execute
    browser automation directly; it tracks state transitions and task context for
    reproducible scoring.
    """

    def __init__(self, headless: bool = True, browser: str = "chromium", viewport: Optional[Dict[str, int]] = None):
        self.headless = headless
        self.browser = browser
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.started = False
        self.state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    async def setup(self) -> bool:
        """Initialize environment state."""
        self.started = True
        self.state = {
            "browser": self.browser,
            "headless": self.headless,
            "viewport": self.viewport,
            "url": None,
        }
        self.history = []
        return True

    async def reset(self) -> bool:
        """Reset environment between tasks/runs."""
        if not self.started:
            await self.setup()
        self.state["url"] = None
        self.history = []
        return True

    async def navigate_to(self, url: str) -> Dict[str, Any]:
        """Track the start URL for task execution."""
        if not url:
            return {"url": "", "status": "skipped"}

        self.state["url"] = url
        event = {"event": "navigate", "url": url, "timestamp": time.time()}
        self.history.append(event)
        return {"url": url, "status": "loaded"}

    async def get_page_state(self) -> Dict[str, Any]:
        """Return current deterministic page state."""
        return {
            "url": self.state.get("url"),
            "browser": self.browser,
            "headless": self.headless,
            "history_size": len(self.history),
        }

    async def cleanup(self) -> None:
        """Tear down the environment."""
        self.started = False
        self.state = {}
        self.history = []


class TaskJudge:
    """Judge whether a task should be counted as successful."""

    @staticmethod
    def _stringify_output(output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output.lower()
        if isinstance(output, (dict, list)):
            return json.dumps(output, ensure_ascii=False).lower()
        return str(output).lower()

    @staticmethod
    def _extract_keywords(task: Task) -> List[str]:
        raw_expected = task.expected_result or ""
        tokens = [token.strip().lower() for token in raw_expected.split(",") if token.strip()]
        metadata_keywords = task.metadata.get("expected_content", [])
        if isinstance(metadata_keywords, list):
            tokens.extend([str(item).strip().lower() for item in metadata_keywords if str(item).strip()])
        return list(dict.fromkeys(tokens))

    @staticmethod
    def _extract_forbidden(task: Task) -> List[str]:
        forbidden = task.metadata.get("forbidden_content", [])
        if not isinstance(forbidden, list):
            return []
        return [str(item).strip().lower() for item in forbidden if str(item).strip()]

    @classmethod
    def judge(
        cls,
        task: Task,
        task_result: A2ATaskResult,
    ) -> Tuple[bool, Optional[str]]:
        """Judge task result with deterministic, explainable rules."""
        if task_result.state != TaskState.COMPLETED:
            if task_result.state == TaskState.CANCELLED:
                return False, "Task cancelled by assessee."
            return False, task_result.error or f"Task state is {task_result.state.value}."

        output_text = cls._stringify_output(task_result.output)
        expected_keywords = cls._extract_keywords(task)
        for keyword in expected_keywords:
            if keyword not in output_text:
                return False, f"Missing expected keyword: {keyword}"

        forbidden_keywords = cls._extract_forbidden(task)
        for keyword in forbidden_keywords:
            if keyword and keyword in output_text:
                return False, f"Found forbidden keyword: {keyword}"

        criteria = task.metadata.get("evaluation_criteria", {})
        if isinstance(criteria, dict):
            output_obj = task_result.output if isinstance(task_result.output, dict) else {}
            for key, expected_value in criteria.items():
                if isinstance(expected_value, bool) and key in output_obj:
                    if bool(output_obj.get(key)) != expected_value:
                        return False, f"Evaluation criteria mismatch for {key}."

        return True, None


class MetricsCalculator:
    """Calculates task-level and aggregate metrics."""

    @staticmethod
    def safe_mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def safe_stdev(values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        return float(statistics.stdev(values))

    @staticmethod
    def weighted_score(metric_values: Dict[str, float], metric_weights: Dict[str, float]) -> float:
        total = 0.0
        for name, value in metric_values.items():
            total += value * metric_weights.get(name, 0.0)
        return round(total, 6)

    @staticmethod
    def calculate_success_rate(results: Sequence[EvaluationResult]) -> float:
        if not results:
            return 0.0
        return round(sum(1 for result in results if result.success) / len(results), 6)

    @staticmethod
    def calculate_efficiency(results: Sequence[EvaluationResult]) -> float:
        if not results:
            return 0.0
        values = [result.metrics.get("efficiency", 0.0) for result in results]
        return round(MetricsCalculator.safe_mean(values), 6)

    @staticmethod
    def calculate_error_recovery(results: Sequence[EvaluationResult]) -> float:
        if not results:
            return 0.0
        values = [result.metrics.get("error_recovery", 0.0) for result in results]
        return round(MetricsCalculator.safe_mean(values), 6)

    @staticmethod
    def consistency_score(run_metric_values: Dict[str, List[float]]) -> float:
        """
        Convert cross-run variance into a score in [0, 1].

        Lower stdev -> higher score.
        """
        stdev_values = [MetricsCalculator.safe_stdev(values) for values in run_metric_values.values()]
        if not stdev_values:
            return 1.0
        avg_stdev = MetricsCalculator.safe_mean(stdev_values)
        return round(max(0.0, 1.0 - avg_stdev), 6)


class WebAgentGreenAgent:
    """Green Agent orchestrator for evaluating web browsing agents."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("agent", {}).get("name", "Web Agent Evaluator")
        self.version = config.get("agent", {}).get("version", "1.0.0")

        env_config = config.get("environment", {})
        self.environment = WebEnvironment(
            headless=env_config.get("headless", True),
            browser=env_config.get("browser", "chromium"),
            viewport=env_config.get("viewport"),
        )
        self.url_overrides = env_config.get("url_overrides", {}) or {}

        a2a_config = config.get("a2a", {}).get("task_manager", {})
        self.poll_interval = float(a2a_config.get("poll_interval", 1.5))
        self.max_retries = int(a2a_config.get("max_retries", 3))
        self.retry_delay = float(a2a_config.get("retry_delay", 1.0))
        self.request_timeout = int(a2a_config.get("timeout", 300))
        self.task_retry_limit = int(a2a_config.get("task_retry_limit", 1))

        eval_config = config.get("evaluation", {})
        self.default_run_repeats = max(1, int(eval_config.get("run_repeats", 1)))
        self.max_concurrency = max(1, int(eval_config.get("max_concurrency", 1)))
        self.metric_weights = self._load_metric_weights(eval_config.get("metrics", []))
        self._validate_metric_weights(self.metric_weights)

        self.metrics_calc = MetricsCalculator()
        self.judge = TaskJudge()
        self.is_running = False

    @staticmethod
    def _load_metric_weights(metric_config: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for item in metric_config:
            name = item.get("name")
            if name:
                weights[name] = float(item.get("weight", 0.0))
        return weights

    @staticmethod
    def _validate_metric_weights(metric_weights: Dict[str, float]) -> None:
        if not metric_weights:
            raise ValueError("evaluation.metrics must define at least one metric weight.")
        total_weight = sum(metric_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Metric weights must sum to 1.0, got {total_weight:.6f}")

    @staticmethod
    def _extract_used_steps(result: A2ATaskResult) -> Optional[int]:
        for key in ("steps", "step_count", "num_steps"):
            value = result.metrics.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                return int(value)
        return None

    @staticmethod
    def _task_efficiency(task: Task, execution_time: float, result: A2ATaskResult) -> float:
        timeout = max(1, int(task.timeout))
        time_score = max(0.0, 1.0 - (execution_time / timeout))

        max_steps = int(task.metadata.get("max_steps", 20))
        used_steps = WebAgentGreenAgent._extract_used_steps(result)
        if used_steps is None:
            step_score = 0.5
        else:
            step_score = max(0.0, 1.0 - (used_steps / max(1, max_steps)))

        return round((0.7 * time_score) + (0.3 * step_score), 6)

    @staticmethod
    def _error_recovery(success: bool, attempts: int, task_trace: Sequence[Dict[str, Any]]) -> float:
        if success and attempts == 1:
            return 1.0
        if success and attempts > 1:
            return 0.8
        has_retry = any(item.get("event") == "retry" for item in task_trace)
        if has_retry:
            return 0.3
        return 0.0

    async def initialize(self) -> bool:
        """Initialize the Green Agent runtime."""
        try:
            await self.environment.setup()
            self.is_running = True
            return True
        except Exception:
            logger.exception("Failed to initialize Green Agent.")
            self.is_running = False
            return False

    async def shutdown(self) -> None:
        """Shutdown the Green Agent runtime."""
        await self.environment.cleanup()
        self.is_running = False

    async def reset(self) -> None:
        """Reset environment state between assessments."""
        await self.environment.reset()

    async def prepare_environment(self, task: Task) -> bool:
        """Prepare environment and start URL for a task."""
        await self.environment.reset()
        target_url = self._resolve_target_url(task.target_url)
        if target_url:
            await self.environment.navigate_to(target_url)
        return True

    def _resolve_target_url(self, url: str) -> str:
        """Resolve placeholder domains to deploy-time URLs."""
        if not url:
            return ""
        for source, target in self.url_overrides.items():
            if source and target and source in url:
                return url.replace(source, target)
        return url

    async def send_task(self, task: Task, assessee_agent_url: str) -> A2ATaskResult:
        """Send one task via A2A and wait for result."""
        instruction = task.metadata.get("instruction") or task.description
        a2a_task = create_a2a_task(task, instruction=instruction)

        async with A2AClient(
            agent_url=assessee_agent_url,
            timeout=min(task.timeout, self.request_timeout),
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        ) as client:
            return await client.send_task(
                a2a_task,
                wait_for_result=True,
                poll_interval=self.poll_interval,
            )

    async def evaluate_single_task(self, task: Task, assessee_agent_url: str, run_id: int = 1) -> EvaluationResult:
        """Evaluate one task with retry-aware tracing."""
        trace: List[Dict[str, Any]] = []
        await self.prepare_environment(task)
        start = time.perf_counter()

        last_result: Optional[A2ATaskResult] = None
        error_message: Optional[str] = None
        attempts = 0

        for attempt in range(1, self.task_retry_limit + 1):
            attempts = attempt
            try:
                trace.append({"event": "attempt_start", "attempt": attempt, "timestamp": time.time()})
                task_result = await self.send_task(task, assessee_agent_url)
                last_result = task_result
                trace.append(
                    {
                        "event": "attempt_end",
                        "attempt": attempt,
                        "state": task_result.state.value,
                        "timestamp": time.time(),
                    }
                )
                if task_result.state == TaskState.COMPLETED:
                    break
                error_message = task_result.error or f"Assessee returned {task_result.state.value}"
                if attempt < self.task_retry_limit:
                    trace.append({"event": "retry", "attempt": attempt + 1, "timestamp": time.time()})
            except asyncio.TimeoutError:
                error_message = "A2A timeout"
                trace.append({"event": "timeout", "attempt": attempt, "timestamp": time.time()})
                if attempt < self.task_retry_limit:
                    trace.append({"event": "retry", "attempt": attempt + 1, "timestamp": time.time()})
            except Exception as exc:
                error_message = str(exc)
                trace.append({"event": "exception", "attempt": attempt, "error": str(exc), "timestamp": time.time()})
                if attempt < self.task_retry_limit:
                    trace.append({"event": "retry", "attempt": attempt + 1, "timestamp": time.time()})

        execution_time = time.perf_counter() - start

        if last_result is None:
            metrics = {
                "success_rate": 0.0,
                "efficiency": 0.0,
                "error_recovery": self._error_recovery(False, attempts, trace),
            }
            return EvaluationResult(
                task_id=task.id,
                run_id=run_id,
                status=TaskStatus.TIMEOUT if "timeout" in (error_message or "").lower() else TaskStatus.FAILED,
                success=False,
                execution_time=execution_time,
                metrics=metrics,
                attempts=attempts,
                error_message=error_message,
                trace=trace,
            )

        success, judge_reason = self.judge.judge(task, last_result)
        if not success and judge_reason and not error_message:
            error_message = judge_reason

        if last_result.state == TaskState.CANCELLED:
            status = TaskStatus.CANCELLED
        elif last_result.state == TaskState.FAILED:
            status = TaskStatus.FAILED
        elif success:
            status = TaskStatus.SUCCESS
        else:
            status = TaskStatus.FAILED

        task_metrics = {
            "success_rate": 1.0 if success else 0.0,
            "efficiency": self._task_efficiency(task, execution_time, last_result),
            "error_recovery": self._error_recovery(success, attempts, trace),
        }

        return EvaluationResult(
            task_id=task.id,
            run_id=run_id,
            status=status,
            success=success,
            execution_time=execution_time,
            metrics=task_metrics,
            attempts=attempts,
            error_message=error_message,
            trace=trace,
            raw_output=last_result.output,
            raw_metrics=last_result.metrics,
        )

    @staticmethod
    def _normalize_tasks(tasks: Iterable[Union[Task, Dict[str, Any]]]) -> List[Task]:
        normalized: List[Task] = []
        for task in tasks:
            if isinstance(task, Task):
                normalized.append(task)
            elif isinstance(task, dict):
                normalized.append(Task.from_dict(task))
            else:
                raise TypeError(f"Unsupported task type: {type(task)}")
        return normalized

    async def _evaluate_run(self, tasks: Sequence[Task], assessee_agent_url: str, run_id: int) -> List[EvaluationResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_task(task: Task) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate_single_task(task, assessee_agent_url, run_id=run_id)

        return await asyncio.gather(*[run_task(task) for task in tasks])

    def _aggregate_metrics(self, results: Sequence[EvaluationResult]) -> Dict[str, float]:
        return {
            "success_rate": self.metrics_calc.calculate_success_rate(results),
            "efficiency": self.metrics_calc.calculate_efficiency(results),
            "error_recovery": self.metrics_calc.calculate_error_recovery(results),
        }

    async def evaluate(
        self,
        tasks: List[Union[Task, Dict[str, Any]]],
        assessee_agent_url: str,
        run_repeats: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate assessee agent over task set with optional multi-run reliability stats.
        """
        if not self.is_running:
            raise RuntimeError("Green Agent is not initialized. Call initialize() first.")

        normalized_tasks = self._normalize_tasks(tasks)
        if not normalized_tasks:
            raise ValueError("Task list is empty.")

        repeats = max(1, int(run_repeats or self.default_run_repeats))
        run_summaries: List[Dict[str, Any]] = []
        all_results: List[EvaluationResult] = []

        for run_id in range(1, repeats + 1):
            logger.info("Starting run %s/%s for %s tasks.", run_id, repeats, len(normalized_tasks))
            run_results = await self._evaluate_run(normalized_tasks, assessee_agent_url, run_id=run_id)
            all_results.extend(run_results)
            run_metrics = self._aggregate_metrics(run_results)
            run_weighted = self.metrics_calc.weighted_score(run_metrics, self.metric_weights)
            run_summaries.append(
                {
                    "run_id": run_id,
                    "total_tasks": len(run_results),
                    "successful_tasks": sum(1 for result in run_results if result.success),
                    "metrics": run_metrics,
                    "weighted_score": run_weighted,
                }
            )

        metric_names = sorted(self.metric_weights.keys())
        per_run_metric_values = {
            name: [summary["metrics"].get(name, 0.0) for summary in run_summaries] for name in metric_names
        }
        aggregate_metrics = {
            name: round(self.metrics_calc.safe_mean(values), 6) for name, values in per_run_metric_values.items()
        }
        metric_std = {name: round(self.metrics_calc.safe_stdev(values), 6) for name, values in per_run_metric_values.items()}
        weighted_score = self.metrics_calc.weighted_score(aggregate_metrics, self.metric_weights)

        return {
            "agent_name": self.name,
            "version": self.version,
            "assessee_agent_url": assessee_agent_url,
            "run_repeats": repeats,
            "total_tasks": len(normalized_tasks),
            "successful_tasks": sum(1 for result in all_results if result.success),
            "metrics": aggregate_metrics,
            "metric_std": metric_std,
            "consistency_score": self.metrics_calc.consistency_score(per_run_metric_values),
            "weighted_score": weighted_score,
            "run_summaries": run_summaries,
            "detailed_results": [result.to_dict() for result in all_results],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def get_agent_info(self) -> Dict[str, Any]:
        """Get metadata used by controller/platform registration."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.config.get("agent", {}).get("description", ""),
            "capabilities": ["web_evaluation", "benchmarking", "multi_run_reliability"],
            "protocols": ["A2A", "MCP"],
            "metric_weights": self.metric_weights,
            "max_concurrency": self.max_concurrency,
            "default_run_repeats": self.default_run_repeats,
        }


async def main() -> None:
    """Manual CLI entrypoint for local smoke test."""
    import yaml

    with open("config/config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    agent = WebAgentGreenAgent(config)
    ready = await agent.initialize()
    if not ready:
        raise RuntimeError("Failed to initialize agent.")

    try:
        demo_tasks = [
            Task(
                id="task_001",
                description="Demo task",
                target_url="https://example.org",
                expected_result="completed",
                metadata={"instruction": "Complete the demo task and report status=completed."},
            )
        ]
        result = await agent.evaluate(demo_tasks, "http://localhost:8001")
        print(json.dumps(result, indent=2))
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
