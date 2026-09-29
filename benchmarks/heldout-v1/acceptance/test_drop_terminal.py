import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class DropTerminalTests(unittest.TestCase):
    def test_done_and_failed(self):
        q = JobQueue()
        q.submit("a", {"tags": [1]})
        q.submit("b", {})
        q.submit("c", {})
        q.claim()
        q.ack("a")
        q.claim()
        q.fail("b")
        removed = q.drop_terminal("a")
        self.assertEqual(removed, {"id": "a", "state": "done", "attempts": 1,
                                   "payload": {"tags": [1]}})
        removed["payload"]["tags"].append(2)
        self.assertNotIn("a", q.jobs)
        self.assertEqual(q.drop_terminal("b")["state"], "failed")
        self.assertEqual(q.pending, ["c"])

    def test_invalid_drops(self):
        q = JobQueue()
        q.submit("a", {})
        q.submit("b", {})
        q.claim()
        for name, error in (("a", ValueError), ("b", ValueError), ("x", KeyError)):
            with self.assertRaises(error):
                q.drop_terminal(name)
        self.assertEqual(list(q.jobs), ["a", "b"])

    def test_api(self):
        api = QueueAPI()
        api.handle({"action": "submit", "id": "a"})
        api.handle({"action": "claim"})
        api.handle({"action": "ack", "id": "a"})
        self.assertEqual(api.handle({"action": "drop_terminal", "id": "a"}),
                         {"id": "a", "state": "done", "attempts": 1, "payload": {}})


if __name__ == "__main__":
    unittest.main()
