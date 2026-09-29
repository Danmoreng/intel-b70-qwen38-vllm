import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class MovePendingTests(unittest.TestCase):
    def test_reorder(self):
        q = JobQueue()
        for name in ("a", "b", "c"):
            q.submit(name, {})
        self.assertIs(q.move_pending("c", "a"), q.jobs["c"])
        self.assertEqual(q.pending, ["c", "a", "b"])
        self.assertEqual([q.claim().job_id for _ in range(3)], ["c", "a", "b"])

    def test_invalid_moves_leave_order(self):
        q = JobQueue()
        for name in ("a", "b", "c"):
            q.submit(name, {})
        q.claim()
        for old, before, error in (("a", "b", ValueError), ("b", "a", ValueError),
                                   ("b", "b", ValueError), ("x", "b", KeyError)):
            with self.assertRaises(error):
                q.move_pending(old, before)
            self.assertEqual(q.pending, ["b", "c"])

    def test_api(self):
        api = QueueAPI()
        for name in ("a", "b"):
            api.handle({"action": "submit", "id": name})
        self.assertEqual(api.handle({"action": "move_pending", "id": "b", "before_id": "a"}),
                         {"id": "b", "state": "pending", "pending": ["b", "a"]})


if __name__ == "__main__":
    unittest.main()
