import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class BaselineTests(unittest.TestCase):
    def test_lifecycle(self):
        queue = JobQueue()
        queue.submit("a", {"x": 1})
        self.assertEqual(queue.claim().job_id, "a")
        self.assertEqual(queue.ack("a").state, "done")
        self.assertIsNone(queue.claim())

    def test_json_adapter(self):
        api = QueueAPI()
        self.assertEqual(api.handle({"action": "submit", "id": "a"})["state"], "pending")
        self.assertEqual(api.handle({"action": "claim"})["id"], "a")


if __name__ == "__main__":
    unittest.main()
