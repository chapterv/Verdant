"""Service and controller entrypoint for the Green Agent."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from aiohttp import ClientSession, ClientTimeout
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Task, WebAgentGreenAgent


class TaskPayload(BaseModel):
    """HTTP/RPC task payload."""

    id: str
    description: str
    target_url: str
    expected_result: str
    timeout: int = 300
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_task(self) -> Task:
        return Task(
            id=self.id,
            description=self.description,
            target_url=self.target_url,
            expected_result=self.expected_result,
            timeout=self.timeout,
            metadata=self.metadata,
        )


class AssessmentRunRequest(BaseModel):
    """Assessment run request."""

    assessee_agent_url: str
    tasks: List[TaskPayload]
    run_repeats: Optional[int] = None


class ProxyRequest(BaseModel):
    """Controller proxy request."""

    method: str = "GET"
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    json_body: Optional[Any] = None
    timeout_seconds: int = 20


class GreenAgentRuntime:
    """In-process runtime and assessment registry."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.agent = WebAgentGreenAgent(self.config)

        self.assessments: Dict[str, Dict[str, Any]] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    def _load_config(self) -> Dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    async def initialize(self) -> bool:
        return await self.agent.initialize()

    async def shutdown(self) -> None:
        for assessment_id, task in list(self.running_jobs.items()):
            if not task.done():
                task.cancel()
            self.assessments.setdefault(assessment_id, {})
            self.assessments[assessment_id]["state"] = "cancelled"
            self.assessments[assessment_id]["error"] = "Cancelled due to shutdown"
        self.running_jobs.clear()
        await self.agent.shutdown()

    async def restart(self) -> bool:
        await self.agent.shutdown()
        self.config = self._load_config()
        self.agent = WebAgentGreenAgent(self.config)
        return await self.agent.initialize()

    async def reset(self) -> None:
        await self.agent.reset()

    async def run_assessment_sync(self, request: AssessmentRunRequest) -> Dict[str, Any]:
        tasks = [item.to_task() for item in request.tasks]
        return await self.agent.evaluate(
            tasks=tasks,
            assessee_agent_url=request.assessee_agent_url,
            run_repeats=request.run_repeats,
        )

    async def submit_assessment(self, request: AssessmentRunRequest) -> str:
        assessment_id = f"asmt_{uuid.uuid4().hex[:12]}"

        async with self._lock:
            self.assessments[assessment_id] = {
                "assessment_id": assessment_id,
                "state": "submitted",
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request": request.model_dump(),
                "result": None,
                "error": None,
            }

        async def _runner() -> None:
            async with self._lock:
                self.assessments[assessment_id]["state"] = "running"
                self.assessments[assessment_id]["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            try:
                result = await self.run_assessment_sync(request)
                async with self._lock:
                    self.assessments[assessment_id]["state"] = "completed"
                    self.assessments[assessment_id]["result"] = result
                    self.assessments[assessment_id]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            except asyncio.CancelledError:
                async with self._lock:
                    self.assessments[assessment_id]["state"] = "cancelled"
                    self.assessments[assessment_id]["error"] = "Assessment cancelled"
                    self.assessments[assessment_id]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                raise
            except Exception as exc:  # pragma: no cover - defensive path
                async with self._lock:
                    self.assessments[assessment_id]["state"] = "failed"
                    self.assessments[assessment_id]["error"] = str(exc)
                    self.assessments[assessment_id]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            finally:
                self.running_jobs.pop(assessment_id, None)

        task = asyncio.create_task(_runner())
        self.running_jobs[assessment_id] = task
        return assessment_id

    async def get_assessment(self, assessment_id: str) -> Dict[str, Any]:
        async with self._lock:
            if assessment_id not in self.assessments:
                raise KeyError(assessment_id)
            return self.assessments[assessment_id]

    async def cancel_assessment(self, assessment_id: str) -> bool:
        job = self.running_jobs.get(assessment_id)
        if not job:
            return False
        job.cancel()
        return True


class RPCError(Exception):
    """JSON-RPC error wrapper."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def rpc_ok(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def rpc_error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def create_app(config_path: str = "config/config.yaml") -> FastAPI:
    runtime = GreenAgentRuntime(config_path=config_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ok = await runtime.initialize()
        if not ok:
            raise RuntimeError("Failed to initialize Green Agent runtime")
        try:
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(title="Web Agent Green Service", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok" if runtime.agent.is_running else "degraded",
            "agent_running": runtime.agent.is_running,
            "running_jobs": len(runtime.running_jobs),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    @app.get("/agent/info")
    async def agent_info() -> Dict[str, Any]:
        return runtime.agent.get_agent_info()

    @app.get("/controller/state")
    async def controller_state() -> Dict[str, Any]:
        return {
            "agent_running": runtime.agent.is_running,
            "running_jobs": list(runtime.running_jobs.keys()),
            "assessment_count": len(runtime.assessments),
            "config_path": str(runtime.config_path),
        }

    @app.post("/controller/reset")
    async def controller_reset() -> Dict[str, Any]:
        await runtime.reset()
        return {"status": "ok", "message": "Environment reset complete"}

    @app.post("/controller/restart")
    async def controller_restart() -> Dict[str, Any]:
        ok = await runtime.restart()
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to restart runtime")
        return {"status": "ok", "message": "Runtime restarted"}

    @app.post("/controller/proxy")
    async def controller_proxy(request: ProxyRequest) -> Dict[str, Any]:
        timeout = ClientTimeout(total=request.timeout_seconds)
        async with ClientSession(timeout=timeout) as session:
            async with session.request(
                method=request.method.upper(),
                url=request.url,
                headers=request.headers,
                json=request.json_body,
            ) as response:
                body = await response.text()
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "body": body,
                }

    @app.post("/assessments/run")
    async def run_assessment(request: AssessmentRunRequest, sync: bool = False) -> Dict[str, Any]:
        if sync:
            result = await runtime.run_assessment_sync(request)
            return {"state": "completed", "result": result}

        assessment_id = await runtime.submit_assessment(request)
        return {"assessment_id": assessment_id, "state": "submitted"}

    @app.get("/assessments/{assessment_id}")
    async def get_assessment(assessment_id: str) -> Dict[str, Any]:
        try:
            return await runtime.get_assessment(assessment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Assessment not found: {assessment_id}") from exc

    @app.post("/assessments/{assessment_id}/cancel")
    async def cancel_assessment(assessment_id: str) -> Dict[str, Any]:
        cancelled = await runtime.cancel_assessment(assessment_id)
        if not cancelled:
            raise HTTPException(status_code=404, detail=f"Running assessment not found: {assessment_id}")
        return {"assessment_id": assessment_id, "state": "cancelled"}

    @app.post("/rpc")
    async def rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params", {})

        try:
            if method == "agent/info":
                return rpc_ok(request_id, runtime.agent.get_agent_info())

            if method == "assessments/run":
                request = AssessmentRunRequest.model_validate(params)
                assessment_id = await runtime.submit_assessment(request)
                return rpc_ok(request_id, {"assessment_id": assessment_id, "state": "submitted"})

            if method == "assessments/get":
                assessment_id = params.get("assessment_id")
                if not assessment_id:
                    raise RPCError(code=-32602, message="assessment_id is required")
                result = await runtime.get_assessment(assessment_id)
                return rpc_ok(request_id, result)

            if method == "assessments/cancel":
                assessment_id = params.get("assessment_id")
                if not assessment_id:
                    raise RPCError(code=-32602, message="assessment_id is required")
                cancelled = await runtime.cancel_assessment(assessment_id)
                if not cancelled:
                    raise RPCError(code=-32000, message=f"Assessment not running: {assessment_id}")
                return rpc_ok(request_id, {"assessment_id": assessment_id, "state": "cancelled"})

            raise RPCError(code=-32601, message=f"Method not found: {method}")
        except RPCError as exc:
            return rpc_error(request_id, exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - defensive path
            return rpc_error(request_id, -32000, str(exc))

    return app


app = create_app()
