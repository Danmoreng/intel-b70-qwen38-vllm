import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class PriorityRetryCancel(unittest.TestCase):
    def test_priority_and_stable_ties(self):
        queue = JobQueue()
        for name, priority in [("low", 1), ("high-a", 3), ("high-b", 3), ("middle", 2)]:
            queue.submit(name, {}, priority)
        self.assertEqual([queue.claim().job_id for _ in range(4)],
                         ["high-a", "high-b", "middle", "low"])

    def test_retry_moves_to_end_of_priority(self):
        queue = JobQueue(max_attempts=2)
        queue.submit("a", {}, 4)
        self.assertEqual(queue.claim().job_id, "a")
        queue.submit("b", {}, 4)
        self.assertEqual(queue.fail("a").state, "pending")
        self.assertEqual(queue.claim().job_id, "b")
        self.assertEqual(queue.claim().job_id, "a")
        self.assertEqual(queue.status("a").attempts, 2)
        self.assertEqual(queue.fail("a").state, "failed")
        self.assertIsNone(queue.claim())

    def test_cancel_never_reappears(self):
        queue = JobQueue()
        queue.submit("a", {})
        queue.submit("b", {})
        self.assertEqual(queue.cancel("a").state, "cancelled")
        self.assertEqual(queue.claim().job_id, "b")
        self.assertEqual(queue.cancel("b").state, "cancelled")
        self.assertIsNone(queue.claim())
        with self.assertRaises(ValueError):
            queue.cancel("a")

    def test_input_errors_and_api(self):
        with self.assertRaises(ValueError):
            JobQueue(0)
        queue = JobQueue()
        for name, payload in [("", {}), (12, {}), ("x", [])]:
            with self.assertRaises((ValueError, TypeError)):
                queue.submit(name, payload)
        with self.assertRaises(KeyError):
            queue.status("missing")
        api = QueueAPI()
        api.handle({"action": "submit", "id": "x"})
        self.assertEqual(api.handle({"action": "status", "id": "x"}),
                         {"id": "x", "state": "pending", "attempts": 0})
        self.assertEqual(api.handle({"action": "cancel", "id": "x"})["state"],
                         "cancelled")


if __name__ == "__main__":
    unittest.main()
