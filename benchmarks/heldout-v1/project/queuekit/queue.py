"""A deliberately small baseline queue; benchmark tasks extend its contract."""

from .models import Job


class JobQueue:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.pending: list[str] = []

    def submit(self, job_id: str, payload: dict, priority: int = 0) -> Job:
        if job_id in self.jobs:
            raise ValueError("duplicate job ID")
        job = Job(job_id, dict(payload), priority)
        self.jobs[job_id] = job
        self.pending.append(job_id)
        return job

    def claim(self) -> Job | None:
        if not self.pending:
            return None
        job = self.jobs[self.pending.pop(0)]
        job.state = "running"
        job.attempts += 1
        return job

    def ack(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        if job.state != "running":
            raise ValueError("job is not running")
        job.state = "done"
        return job

    def fail(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        if job.state != "running":
            raise ValueError("job is not running")
        job.state = "failed"
        return job
