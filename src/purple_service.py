"""FastAPI service for the Verdant Purple assessee agent."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException

from .a2a_protocol import TaskState
from .purple_agent import PurpleAgent, PurpleTask


class PurpleRuntime:
    """In-memory runtime for Purple task execution."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.agent = PurpleAgent(self.config)
        self.task_store: Dict[str, Dict[str, Any]] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    def _load_config(self) -> Dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    async def initialize(self) -> bool:
        return await self.agent.initialize()

    async def shutdown(self) -> None:
        for task_id, job in list(self.running_jobs.items()):
            if not job.done():
                job.cancel()
            self.task_store.setdefault(task_id, {})
            self.task_store[task_id]["state"] = TaskState.CANCELLED.value
            self.task_store[task_id]["error"] = "Cancelled due to shutdown"
        if self.running_jobs:
            await asyncio.gather(*self.running_jobs.values(), return_exceptions=True)
        self.running_jobs.clear()
        await self.agent.shutdown()

    async def submit_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        task = PurpleTask.from_rpc_payload(task_payload)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        async with self._lock:
            self.task_store[task.id] = {
                "task_id": task.id,
                "state": TaskState.SUBMITTED.value,
                "output": None,
                "error": None,
                "artifacts": [],
                "metrics": {},
                "trace": [],
                "created_at": now,
                "updated_at": now,
            }

        async def _runner() -> None:
            async with self._lock:
                self.task_store[task.id]["state"] = TaskState.WORKING.value
                self.task_store[task.id]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            try:
                result = await self.agent.execute_task(task)
                async with self._lock:
                    self.task_store[task.id]["state"] = result.state.value
                    self.task_store[task.id]["output"] = result.output
                    self.task_store[task.id]["error"] = result.error
                    self.task_store[task.id]["artifacts"] = result.artifacts
                    self.task_store[task.id]["metrics"] = result.metrics
                    self.task_store[task.id]["trace"] = result.trace
                    self.task_store[task.id]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            except asyncio.CancelledError:
                async with self._lock:
                    self.task_store[task.id]["state"] = TaskState.CANCELLED.value
                    self.task_store[task.id]["error"] = "Task cancelled"
                    self.task_store[task.id]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                raise
            except Exception as exc:  # pragma: no cover - defensive path
                async with self._lock:
                    self.task_store[task.id]["state"] = TaskState.FAILED.value
                    self.task_store[task.id]["error"] = str(exc)
                    self.task_store[task.id]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            finally:
                self.running_jobs.pop(task.id, None)

        job = asyncio.create_task(_runner())
        self.running_jobs[task.id] = job
        return {"task_id": task.id, "state": TaskState.SUBMITTED.value}

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        async with self._lock:
            if task_id not in self.task_store:
                raise KeyError(task_id)
            task_data = self.task_store[task_id]
            return {
                "task_id": task_id,
                "state": task_data["state"],
                "output": task_data.get("output"),
                "error": task_data.get("error"),
                "artifacts": task_data.get("artifacts", []),
                "metrics": task_data.get("metrics", {}),
            }

    async def get_trace(self, task_id: str) -> Dict[str, Any]:
        async with self._lock:
            if task_id not in self.task_store:
                raise KeyError(task_id)
            task_data = self.task_store[task_id]
            return {
                "task_id": task_id,
                "state": task_data["state"],
                "trace": task_data.get("trace", []),
                "updated_at": task_data.get("updated_at"),
            }

    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        job = self.running_jobs.get(task_id)
        if not job:
            if task_id not in self.task_store:
                raise KeyError(task_id)
            return {"task_id": task_id, "state": self.task_store[task_id]["state"]}

        job.cancel()
        return {"task_id": task_id, "state": TaskState.CANCELLED.value}


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


def create_purple_app(config_path: str = "config/purple_config.yaml") -> FastAPI:
    runtime = PurpleRuntime(config_path=config_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ok = await runtime.initialize()
        if not ok:
            raise RuntimeError("Failed to initialize Purple runtime")
        try:
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(title="Verdant Purple Service", version="1.0.0", lifespan=lifespan)

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

    @app.get("/traces/{task_id}")
    async def get_trace(task_id: str) -> Dict[str, Any]:
        try:
            return await runtime.get_trace(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}") from exc

    @app.post("/rpc")
    async def rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params", {}) or {}

        try:
            if method == "agent/info":
                return rpc_ok(request_id, runtime.agent.get_agent_info())

            if method == "tasks/send":
                task_payload = params.get("task")
                if not isinstance(task_payload, dict):
                    raise RPCError(code=-32602, message="params.task is required")
                result = await runtime.submit_task(task_payload)
                return rpc_ok(request_id, result)

            if method == "tasks/get":
                task_id = params.get("task_id")
                if not task_id:
                    raise RPCError(code=-32602, message="task_id is required")
                result = await runtime.get_task(str(task_id))
                return rpc_ok(request_id, result)

            if method == "tasks/cancel":
                task_id = params.get("task_id")
                if not task_id:
                    raise RPCError(code=-32602, message="task_id is required")
                result = await runtime.cancel_task(str(task_id))
                return rpc_ok(request_id, result)

            raise RPCError(code=-32601, message=f"Method not found: {method}")
        except KeyError as exc:
            return rpc_error(request_id, -32000, f"Task not found: {exc.args[0]}")
        except RPCError as exc:
            return rpc_error(request_id, exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - defensive path
            return rpc_error(request_id, -32000, str(exc))

    return app


app = create_purple_app()
