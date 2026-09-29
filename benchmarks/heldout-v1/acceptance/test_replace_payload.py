import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class ReplacePayloadTests(unittest.TestCase):
    def test_deep_copy(self):
        q = JobQueue()
        q.submit("a", {})
        input_payload = {"nested": [1]}
        job = q.replace_payload("a", input_payload)
        self.assertIs(job, q.jobs["a"])
        input_payload["nested"].append(2)
        self.assertEqual(job.payload, {"nested": [1]})

    def test_invalid_state_and_input(self):
        q = JobQueue()
        q.submit("a", {})
        q.submit("b", {})
        q.claim()
        for name, payload, error in (("a", {}, ValueError), ("b", [], ValueError),
                                     ("x", {}, KeyError)):
            with self.assertRaises(error):
                q.replace_payload(name, payload)
        self.assertEqual(q.jobs["b"].payload, {})

    def test_api_response_detached(self):
        api = QueueAPI()
        api.handle({"action": "submit", "id": "a"})
        result = api.handle({"action": "replace_payload", "id": "a",
                             "payload": {"nested": [1]}})
        self.assertEqual(result, {"id": "a", "state": "pending",
                                  "payload": {"nested": [1]}})
        result["payload"]["nested"].append(2)
        self.assertEqual(api.queue.jobs["a"].payload, {"nested": [1]})


if __name__ == "__main__":
    unittest.main()
