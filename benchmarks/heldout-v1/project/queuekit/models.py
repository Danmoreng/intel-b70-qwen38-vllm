"""The queue's public job state."""

from dataclasses import dataclass


@dataclass
class Job:
    job_id: str
    payload: dict
    priority: int = 0
    attempts: int = 0
    state: str = "pending"
