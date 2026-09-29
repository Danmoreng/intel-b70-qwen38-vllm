import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class ClaimMatchingTests(unittest.TestCase):
    def test_first_match_and_preserved_order(self):
        q = JobQueue()
        q.submit("a", {"zone": "west"})
        q.submit("b", {"zone": "east"})
        q.submit("c", {"zone": "east"})
        self.assertEqual(q.claim_matching("zone", "east").job_id, "b")
        self.assertEqual(q.pending, ["a", "c"])
        self.assertEqual(q.jobs["b"].attempts, 1)
        self.assertIsNone(q.claim_matching("zone", "north"))

    def test_invalid_key(self):
        q = JobQueue()
        q.submit("a", {"k": 1})
        for key in ("", None, 1):
            with self.assertRaises(ValueError):
                q.claim_matching(key, 1)
        self.assertEqual(q.pending, ["a"])

    def test_api(self):
        api = QueueAPI()
        api.handle({"action": "submit", "id": "a", "payload": {"k": 0}})
        self.assertEqual(api.handle({"action": "claim_matching", "key": "k", "value": 0}),
                         {"id": "a", "state": "running", "attempts": 1})
        self.assertEqual(api.handle({"action": "claim_matching", "key": "k", "value": 0}),
                         {"id": None})


if __name__ == "__main__":
    unittest.main()
