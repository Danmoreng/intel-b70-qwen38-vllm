import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class AckManyTests(unittest.TestCase):
    def test_commit_order(self):
        q = JobQueue()
        for name in ("a", "b", "c"):
            q.submit(name, {})
            q.claim()
        self.assertEqual([j.job_id for j in q.ack_many(["c", "a"])], ["c", "a"])
        self.assertEqual([q.jobs[x].state for x in ("a", "b", "c")],
                         ["done", "running", "done"])

    def test_atomic_validation(self):
        q = JobQueue()
        q.submit("a", {})
        q.submit("b", {})
        q.claim()
        for ids, error in ((["a", "b"], ValueError), (["a", "x"], KeyError),
                           (["a", "a"], ValueError), ([], ValueError),
                           (["a", 1], ValueError)):
            with self.assertRaises(error):
                q.ack_many(ids)
            self.assertEqual(q.jobs["a"].state, "running")
            self.assertEqual(q.jobs["b"].state, "pending")

    def test_api(self):
        api = QueueAPI()
        api.handle({"action": "submit", "id": "a"})
        api.handle({"action": "claim"})
        self.assertEqual(api.handle({"action": "ack_many", "ids": ["a"]}),
                         {"jobs": [{"id": "a", "state": "done"}]})


if __name__ == "__main__":
    unittest.main()
