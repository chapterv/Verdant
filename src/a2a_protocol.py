"""
A2A Protocol Client Implementation
Agent-to-Agent communication protocol for AgentBeats platform
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from aiohttp import ClientSession, ClientTimeout, TCPConnector
import socket

logger = logging.getLogger(__name__)


class A2AMessageType(Enum):
    """A2A Protocol Message Types"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_STATUS = "task_status"
    RESULT = "result"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class TaskState(Enum):
    """Task State in A2A Protocol"""
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class A2AMessage:
    """A2A Protocol Message"""
    message_type: A2AMessageType
    message_id: str
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    content: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class A2ATask:
    """A2A Task Definition"""
    id: str
    description: str
    instruction: str
    expected_output: Optional[str] = None
    timeout: int = 300
    tools: List[str] = field(default_factory=list)
    context: Dict = field(default_factory=dict)


@dataclass
class A2ATaskResult:
    """A2A Task Result"""
    task_id: str
    state: TaskState
    output: Optional[Any] = None
    error: Optional[str] = None
    artifacts: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class A2AClient:
    """
    A2A Protocol Client
    
    Used by Green Agent to communicate with Purple/White Agents
    """
    
    def __init__(
        self, 
        agent_url: str,
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.agent_url = agent_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session: Optional[ClientSession] = None
        self.message_counter = 0
        
        logger.info(f"Initialized A2A Client for {agent_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Establish connection to the agent"""
        timeout = ClientTimeout(total=self.timeout)
        connector = TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            family=socket.AF_INET
        )
        
        self.session = ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        logger.info(f"Connected to agent at {self.agent_url}")
    
    async def disconnect(self):
        """Close connection to the agent"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Disconnected from agent at {self.agent_url}")
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        self.message_counter += 1
        return f"msg_{self.message_counter}_{int(asyncio.get_event_loop().time() * 1000)}"
    
    async def send_task(
        self, 
        task: A2ATask,
        wait_for_result: bool = True,
        poll_interval: float = 2.0
    ) -> A2ATaskResult:
        """
        Send a task to the agent and optionally wait for result
        
        Args:
            task: Task to send
            wait_for_result: Whether to wait for the result
            poll_interval: Interval for polling status (seconds)
            
        Returns:
            A2ATaskResult with the task result
        """
        if not self.session:
            await self.connect()
        
        # Send task request
        request_payload = {
            "jsonrpc": "2.0",
            "id": self._generate_message_id(),
            "method": "tasks/send",
            "params": {
                "task": {
                    "id": task.id,
                    "description": task.description,
                    "instruction": task.instruction,
                    "expectedOutput": task.expected_output,
                    "timeout": task.timeout,
                    "tools": task.tools,
                    "context": task.context
                }
            }
        }
        
        logger.info(f"Sending task {task.id} to agent")
        
        try:
            # Send task with retry logic
            result = await self._send_with_retry(
                f"{self.agent_url}/rpc",
                request_payload
            )
            
            if not wait_for_result:
                return A2ATaskResult(
                    task_id=task.id,
                    state=TaskState.SUBMITTED,
                    output=result
                )
            
            # Poll for result
            return await self._poll_for_result(task.id, poll_interval)
            
        except Exception as e:
            logger.error(f"Failed to send task {task.id}: {e}")
            return A2ATaskResult(
                task_id=task.id,
                state=TaskState.FAILED,
                error=str(e)
            )
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status from the agent"""
        if not self.session:
            await self.connect()
        
        request_payload = {
            "jsonrpc": "2.0",
            "id": self._generate_message_id(),
            "method": "tasks/get",
            "params": {
                "task_id": task_id
            }
        }
        
        try:
            result = await self._send_with_retry(
                f"{self.agent_url}/rpc",
                request_payload
            )
            return result
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return {"error": str(e)}
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if not self.session:
            await self.connect()
        
        request_payload = {
            "jsonrpc": "2.0",
            "id": self._generate_message_id(),
            "method": "tasks/cancel",
            "params": {
                "task_id": task_id
            }
        }
        
        try:
            await self._send_with_retry(
                f"{self.agent_url}/rpc",
                request_payload
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def _poll_for_result(
        self, 
        task_id: str, 
        poll_interval: float,
        max_attempts: Optional[int] = None
    ) -> A2ATaskResult:
        """Poll for task result"""
        if max_attempts is None:
            max_attempts = max(1, int(self.timeout / max(0.1, poll_interval)))
        else:
            max_attempts = int(max_attempts)
        
        for attempt in range(max_attempts):
            status = await self.get_task_status(task_id)
            
            if "error" in status:
                return A2ATaskResult(
                    task_id=task_id,
                    state=TaskState.FAILED,
                    error=status.get("error")
                )
            
            task_state = status.get("result", {}).get("state", "")
            
            if task_state == TaskState.COMPLETED.value:
                return A2ATaskResult(
                    task_id=task_id,
                    state=TaskState.COMPLETED,
                    output=status.get("result", {}).get("output"),
                    artifacts=status.get("result", {}).get("artifacts", []),
                    metrics=status.get("result", {}).get("metrics", {})
                )
            elif task_state == TaskState.FAILED.value:
                return A2ATaskResult(
                    task_id=task_id,
                    state=TaskState.FAILED,
                    error=status.get("result", {}).get("error", "Unknown error")
                )
            
            logger.debug(f"Task {task_id} status: {task_state}, attempt {attempt + 1}/{max_attempts}")
            await asyncio.sleep(poll_interval)
        
        # Timeout
        return A2ATaskResult(
            task_id=task_id,
            state=TaskState.FAILED,
            error="Task polling timeout"
        )
    
    async def _send_with_retry(
        self, 
        url: str, 
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send request with retry logic"""
        if not self.session:
            raise RuntimeError("A2A session is not connected.")

        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json(content_type=None)
                        if "error" in result:
                            raise Exception(f"RPC Error: {result['error']}")
                        return result
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        wait_time = (attempt + 1) * self.retry_delay
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                    else:
                        raise Exception(f"HTTP Error: {response.status}")
                        
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                wait_time = (attempt + 1) * self.retry_delay
                logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        raise Exception(f"Failed after {self.max_retries} attempts: {last_error}")


class A2AServer:
    """
    A2A Protocol Server
    
    Can be used by Purple/White Agent to expose A2A interface
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.tasks: Dict[str, A2ATask] = {}
        self.results: Dict[str, A2ATaskResult] = {}
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
        
        logger.info(f"Initialized A2A Server on {host}:{port}")
    
    def register_handler(
        self,
        method: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    ):
        """Register a method handler"""
        self.handlers[method] = handler
        logger.info(f"Registered handler for method: {method}")
    
    async def handle_request(self, request: Dict) -> Dict:
        """Handle incoming A2A request"""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        # Route to appropriate handler
        if method == "tasks/send":
            return await self._handle_task_send(params, request_id=request_id)
        elif method == "tasks/get":
            return await self._handle_task_get(params, request_id=request_id)
        elif method == "tasks/cancel":
            return await self._handle_task_cancel(params, request_id=request_id)
        elif method in self.handlers:
            response = await self.handlers[method](params)
            if "jsonrpc" not in response:
                response["jsonrpc"] = "2.0"
            response.setdefault("id", request_id)
            return response
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    async def _handle_task_send(self, params: Dict, request_id: Optional[str] = None) -> Dict:
        """Handle task send request"""
        task_data = params.get("task", {})
        task = A2ATask(
            id=task_data.get("id"),
            description=task_data.get("description"),
            instruction=task_data.get("instruction"),
            expected_output=task_data.get("expectedOutput"),
            timeout=task_data.get("timeout", 300),
            tools=task_data.get("tools", []),
            context=task_data.get("context", {})
        )
        
        self.tasks[task.id] = task
        
        # Start task processing (in background)
        asyncio.create_task(self._process_task(task))
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "task_id": task.id,
                "state": TaskState.SUBMITTED.value
            }
        }
    
    async def _handle_task_get(self, params: Dict, request_id: Optional[str] = None) -> Dict:
        """Handle task get request"""
        task_id = params.get("task_id")
        
        if task_id not in self.results:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": f"Task not found: {task_id}"
                }
            }
        
        result = self.results[task_id]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "task_id": task_id,
                "state": result.state.value,
                "output": result.output,
                "error": result.error,
                "artifacts": result.artifacts,
                "metrics": result.metrics
            }
        }
    
    async def _handle_task_cancel(self, params: Dict, request_id: Optional[str] = None) -> Dict:
        """Handle task cancel request"""
        task_id = params.get("task_id")
        
        # Mark task as cancelled
        if task_id in self.tasks:
            self.results[task_id] = A2ATaskResult(
                task_id=task_id,
                state=TaskState.CANCELLED
            )
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "task_id": task_id,
                "state": TaskState.CANCELLED.value
            }
        }
    
    async def _process_task(self, task: A2ATask):
        """Reference task processor for local integration tests."""
        # Update status to working
        self.results[task.id] = A2ATaskResult(
            task_id=task.id,
            state=TaskState.WORKING
        )
        
        try:
            await asyncio.sleep(0.2)
            self.results[task.id] = A2ATaskResult(
                task_id=task.id,
                state=TaskState.COMPLETED,
                output={
                    "message": "Task completed successfully",
                    "status": "completed",
                    "expected_output": task.expected_output,
                },
                metrics={
                    "execution_time": 0.2,
                    "step_count": min(3, max(1, len(task.tools) or 1)),
                },
            )
            
        except Exception as e:
            self.results[task.id] = A2ATaskResult(
                task_id=task.id,
                state=TaskState.FAILED,
                error=str(e)
            )


# Helper function to create A2A task from Task
def create_a2a_task(task, instruction: str = None) -> A2ATask:
    """Convert internal Task to A2A Task"""
    return A2ATask(
        id=task.id,
        description=task.description,
        instruction=instruction or task.description,
        expected_output=task.expected_result,
        timeout=task.timeout,
        context=task.metadata
    )
