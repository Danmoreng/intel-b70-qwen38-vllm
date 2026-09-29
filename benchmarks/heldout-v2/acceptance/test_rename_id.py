import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class RenameIDTests(unittest.TestCase):
    def test_pending_position_and_identity(self):
        q = JobQueue()
        q.submit("a", {})
        obj = q.submit("b", {"x": 1})
        q.submit("c", {})
        self.assertIs(q.rename_id("b", "new"), obj)
        self.assertEqual(q.pending, ["a", "new", "c"])
        self.assertEqual([q.claim().job_id for _ in range(3)], ["a", "new", "c"])

    def test_running_state_and_atomic_errors(self):
        q = JobQueue()
        q.submit("a", {})
        q.submit("b", {})
        q.claim()
        q.rename_id("a", "running-new")
        self.assertEqual(q.jobs["running-new"].state, "running")
        self.assertEqual(q.jobs["running-new"].attempts, 1)
        for old, new, error in (("missing", "x", KeyError),
                                ("b", "running-new", ValueError),
                                ("b", "", ValueError), ("b", "b", ValueError)):
            with self.assertRaises(error):
                q.rename_id(old, new)
        self.assertEqual(q.pending, ["b"])

    def test_api(self):
        api = QueueAPI()
        api.handle({"action": "submit", "id": "a"})
        self.assertEqual(api.handle({"action": "rename_id", "old_id": "a", "new_id": "b"}),
                         {"old_id": "a", "id": "b", "state": "pending"})


if __name__ == "__main__":
    unittest.main()
