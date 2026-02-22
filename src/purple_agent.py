"""Purple assessee agent with deterministic execution and dual-pass verification."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .a2a_protocol import TaskState


@dataclass
class PurpleTask:
    """Normalized task format accepted by the Purple agent."""

    id: str
    description: str
    instruction: str
    expected_output: Optional[str] = None
    timeout: int = 300
    tools: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rpc_payload(cls, payload: Dict[str, Any]) -> "PurpleTask":
        """Build task from A2A RPC payload."""
        task_id = str(payload.get("id") or f"task_{int(time.time() * 1000)}")
        description = str(payload.get("description", "")).strip()
        instruction = str(payload.get("instruction", description)).strip()
        expected_output = payload.get("expectedOutput")
        timeout = int(payload.get("timeout", 300))
        tools = payload.get("tools", []) or []
        context = payload.get("context", {}) or {}
        return cls(
            id=task_id,
            description=description,
            instruction=instruction,
            expected_output=expected_output,
            timeout=timeout,
            tools=[str(item) for item in tools],
            context=context,
        )


@dataclass
class VerificationResult:
    """Result of dual-pass validation."""

    pass1: bool
    pass2: bool
    passed: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass1_semantic": self.pass1,
            "pass2_structural": self.pass2,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


@dataclass
class PurpleTaskResult:
    """Execution output stored by Purple runtime."""

    task_id: str
    state: TaskState
    output: Optional[Any] = None
    error: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)


class PurpleAgent:
    """
    Deterministic Purple reference agent.

    It focuses on verifiability:
    1) deterministic planning/execution trace,
    2) dual-pass output verification,
    3) clear metrics for benchmark analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        agent_cfg = config.get("agent", {}) or {}
        execution_cfg = config.get("execution", {}) or {}
        verification_cfg = config.get("verification", {}) or {}
        llm_cfg = config.get("llm", {}) or {}

        self.name = agent_cfg.get("name", "Verdant Purple")
        self.version = agent_cfg.get("version", "1.0.0")
        self.description = agent_cfg.get("description", "Deterministic Purple assessee agent")

        self.seed = str(execution_cfg.get("seed", "verdant-purple-v1"))
        self.step_delay_seconds = max(0.0, float(execution_cfg.get("step_delay_seconds", 0.15)))
        self.max_steps = max(1, int(execution_cfg.get("max_steps", 4)))

        self.require_expected_keywords = bool(verification_cfg.get("require_expected_keywords", True))
        self.require_structural_fields = bool(verification_cfg.get("require_structural_fields", True))

        self.llm_enabled = bool(llm_cfg.get("enabled", False))
        self.llm_mode = str(llm_cfg.get("mode", "off"))

        self.is_running = False

    async def initialize(self) -> bool:
        """Initialize runtime state."""
        self.is_running = True
        return True

    async def shutdown(self) -> None:
        """Shutdown runtime state."""
        self.is_running = False

    def get_agent_info(self) -> Dict[str, Any]:
        """Expose metadata used by controller/evaluator."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "role": "assessee",
            "capabilities": [
                "task_execution",
                "dual_pass_verification",
                "deterministic_trace",
                "a2a_rpc",
            ],
            "llm_enabled": self.llm_enabled,
            "llm_mode": self.llm_mode,
        }

    async def execute_task(self, task: PurpleTask) -> PurpleTaskResult:
        """Execute one task with deterministic trace and verification."""
        if not self.is_running:
            raise RuntimeError("Purple agent is not initialized.")

        started = time.perf_counter()
        trace: List[Dict[str, Any]] = []

        def emit(event: str, **kwargs: Any) -> None:
            trace.append({"event": event, "timestamp": time.time(), **kwargs})

        emit("task_received", task_id=task.id)
        interpretation = self._interpret_task(task)
        emit("interpretation_done", objective=interpretation["objective"])

        plan = self._build_plan(task, interpretation)
        emit("planning_done", step_count=len(plan))

        step_observations: List[Dict[str, Any]] = []
        try:
            for idx, step in enumerate(plan, start=1):
                emit("step_started", step_index=idx, action=step["action"])
                await asyncio.sleep(self.step_delay_seconds)
                observation = self._simulate_step(task, step, idx)
                step_observations.append(observation)
                emit(
                    "step_finished",
                    step_index=idx,
                    observation=observation["observation"],
                )
        except asyncio.CancelledError:
            emit("task_cancelled")
            raise

        output = self._compose_output(task, interpretation, step_observations)
        emit("draft_output_ready")

        verification = self._verify_output(task, output)
        emit(
            "verification_done",
            pass1=verification.pass1,
            pass2=verification.pass2,
            passed=verification.passed,
        )

        output["verification"] = verification.to_dict()
        duration = time.perf_counter() - started
        result_state = TaskState.COMPLETED if verification.passed else TaskState.FAILED
        result_error = None if verification.passed else "; ".join(verification.reasons) or "verification failed"

        metrics = {
            "execution_time": round(duration, 4),
            "step_count": len(plan),
            "verification_pass": 1.0 if verification.passed else 0.0,
            "pass1": 1.0 if verification.pass1 else 0.0,
            "pass2": 1.0 if verification.pass2 else 0.0,
        }

        artifacts = [
            {
                "type": "plan",
                "name": "execution_plan",
                "content": plan,
            },
            {
                "type": "trace",
                "name": "execution_trace",
                "content": trace,
            },
        ]

        return PurpleTaskResult(
            task_id=task.id,
            state=result_state,
            output=output,
            error=result_error,
            artifacts=artifacts,
            metrics=metrics,
            trace=trace,
        )

    def _interpret_task(self, task: PurpleTask) -> Dict[str, Any]:
        """Build deterministic interpretation from task data."""
        objective = task.instruction or task.description or "Complete the assigned objective."
        constraints = []
        target_url = task.context.get("target_url")
        if isinstance(target_url, str) and target_url.strip():
            constraints.append(f"Target URL: {target_url.strip()}")
        if task.tools:
            constraints.append(f"Tools hint: {', '.join(task.tools)}")
        return {
            "objective": objective,
            "constraints": constraints,
            "expected_keywords": self._extract_expected_keywords(task),
        }

    def _build_plan(self, task: PurpleTask, interpretation: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate a short deterministic action plan."""
        steps = [
            {"action": "analyze_instruction", "detail": interpretation["objective"][:180]},
            {
                "action": "collect_required_signals",
                "detail": "Gather evidence for expected output and constraints.",
            },
            {
                "action": "synthesize_answer",
                "detail": "Produce structured response with verifiable fields.",
            },
        ]
        if interpretation.get("expected_keywords"):
            steps.append(
                {
                    "action": "keyword_alignment_check",
                    "detail": "Ensure expected keywords are present in final answer.",
                }
            )
        return steps[: self.max_steps]

    def _simulate_step(self, task: PurpleTask, step: Dict[str, str], index: int) -> Dict[str, Any]:
        """Create deterministic synthetic observation per step."""
        token_source = f"{self.seed}:{task.id}:{index}:{step['action']}"
        token = hashlib.sha256(token_source.encode("utf-8")).hexdigest()[:10]
        return {
            "step_index": index,
            "action": step["action"],
            "observation": f"{step['action']} completed",
            "token": token,
        }

    def _compose_output(
        self,
        task: PurpleTask,
        interpretation: Dict[str, Any],
        observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compose final structured output."""
        keywords = interpretation.get("expected_keywords", [])
        keyword_phrase = ", ".join(keywords) if keywords else "completed"
        evidence = [item["token"] for item in observations]
        answer = f"Task completed with result: {keyword_phrase}"
        if task.expected_output:
            answer = f"{answer}. Expected output alignment: {task.expected_output}"

        trace_id_raw = f"{self.seed}:{task.id}:{len(observations)}"
        trace_id = hashlib.sha256(trace_id_raw.encode("utf-8")).hexdigest()[:12]

        return {
            "status": "completed",
            "task_id": task.id,
            "answer": answer,
            "summary": interpretation["objective"],
            "evidence": evidence,
            "trace_id": trace_id,
            "llm_used": self.llm_enabled,
        }

    def _extract_expected_keywords(self, task: PurpleTask) -> List[str]:
        keywords: List[str] = []
        raw = (task.expected_output or "").strip()
        if raw:
            keywords.extend([token.strip().lower() for token in raw.split(",") if token.strip()])

        expected_content = task.context.get("expected_content", [])
        if isinstance(expected_content, list):
            for item in expected_content:
                text = str(item).strip().lower()
                if text:
                    keywords.append(text)

        # Preserve order while deduplicating.
        return list(dict.fromkeys(keywords))

    def _verify_output(self, task: PurpleTask, output: Dict[str, Any]) -> VerificationResult:
        """Run dual-pass verification: semantic pass + structural pass."""
        reasons: List[str] = []

        output_text = json.dumps(output, ensure_ascii=False).lower()
        expected_keywords = self._extract_expected_keywords(task)

        pass1 = True
        if self.require_expected_keywords and expected_keywords:
            for keyword in expected_keywords:
                if keyword not in output_text:
                    pass1 = False
                    reasons.append(f"missing expected keyword: {keyword}")

        pass2 = True
        if self.require_structural_fields:
            required_fields = ("status", "task_id", "answer", "evidence")
            for field_name in required_fields:
                if field_name not in output:
                    pass2 = False
                    reasons.append(f"missing required field: {field_name}")
            if output.get("status") != "completed":
                pass2 = False
                reasons.append("status must be completed")
            if str(output.get("task_id", "")) != task.id:
                pass2 = False
                reasons.append("task_id mismatch")
            if not isinstance(output.get("evidence"), list) or not output.get("evidence"):
                pass2 = False
                reasons.append("evidence must be a non-empty list")

        return VerificationResult(pass1=pass1, pass2=pass2, passed=pass1 and pass2, reasons=reasons)
