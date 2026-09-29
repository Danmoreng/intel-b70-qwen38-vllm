"""JSON-like command adapter for the in-memory queue."""

from .queue import JobQueue


class QueueAPI:
    def __init__(self, queue: JobQueue | None = None):
        self.queue = queue or JobQueue()

    def handle(self, command: dict) -> dict:
        action = command["action"]
        if action == "submit":
            job = self.queue.submit(command["id"], command.get("payload", {}),
                                    command.get("priority", 0))
            return {"id": job.job_id, "state": job.state}
        if action == "claim":
            job = self.queue.claim()
            return {"id": job.job_id, "state": job.state} if job else {"id": None}
        if action == "ack":
            job = self.queue.ack(command["id"])
            return {"id": job.job_id, "state": job.state}
        if action == "fail":
            job = self.queue.fail(command["id"])
            return {"id": job.job_id, "state": job.state}
        raise ValueError(f"unknown action: {action}")
