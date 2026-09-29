import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class SubmitManyTests(unittest.TestCase):
    def test_commit_order_and_defaults(self):
        q = JobQueue()
        jobs = q.submit_many([{"id": "a"}, {"id": "b", "payload": {"v": 2}, "priority": 4}])
        self.assertEqual([j.job_id for j in jobs], ["a", "b"])
        self.assertEqual([q.claim().job_id for _ in range(2)], ["a", "b"])
        self.assertEqual(q.jobs["b"].payload, {"v": 2})

    def test_validation_is_atomic(self):
        q = JobQueue()
        q.submit("live", {})
        bad = ([{"id": "a"}, {"id": "live"}], [{"id": "a"}, {"id": "a"}],
               [{"id": "a"}, {"id": ""}], [{"id": "a"}, {"id": "b", "payload": []}],
               [{"id": "a"}, 9], [])
        for items in bad:
            with self.assertRaises(ValueError):
                q.submit_many(items)
            self.assertEqual(list(q.jobs), ["live"])
            self.assertEqual(q.pending, ["live"])

    def test_api(self):
        api = QueueAPI()
        self.assertEqual(api.handle({"action": "submit_many", "items": [{"id": "x"}]}),
                         {"jobs": [{"id": "x", "state": "pending"}]})


if __name__ == "__main__":
    unittest.main()
