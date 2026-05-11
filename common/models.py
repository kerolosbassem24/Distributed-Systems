from dataclasses import dataclass, field
import time

@dataclass
class Request:
    id: int
    query: str
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0

@dataclass
class Response:
    id: int
    result: str
    latency: float
    worker_id: int
    gpu_id: int
    strategy_used: str
    success: bool = True
    error: str = ""
