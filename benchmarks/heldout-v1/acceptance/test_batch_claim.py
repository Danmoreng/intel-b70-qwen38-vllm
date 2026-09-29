import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class BatchClaimTests(unittest.TestCase):
    def test_fifo_and_attempts(self):
        q = JobQueue()
        for name in ("a", "b", "c"):
            q.submit(name, {})
        self.assertEqual([j.job_id for j in q.claim_batch(2)], ["a", "b"])
        self.assertEqual([q.jobs[x].attempts for x in ("a", "b", "c")], [1, 1, 0])
        self.assertEqual([j.job_id for j in q.claim_batch(9)], ["c"])
        self.assertEqual(q.claim_batch(1), [])

    def test_invalid_limit_is_atomic(self):
        q = JobQueue()
        q.submit("a", {})
        for limit in (0, -1, True, 1.5, "2", None):
            with self.assertRaises(ValueError):
                q.claim_batch(limit)
            self.assertEqual(q.pending, ["a"])
            self.assertEqual(q.jobs["a"].attempts, 0)

    def test_api(self):
        api = QueueAPI()
        api.handle({"action": "submit", "id": "a"})
        result = api.handle({"action": "batch_claim", "limit": 3})
        self.assertEqual(result, {"jobs": [{"id": "a", "state": "running", "attempts": 1}]})


if __name__ == "__main__":
    unittest.main()
