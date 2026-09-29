import copy
import json
import unittest

from queuekit.api import QueueAPI
from queuekit.queue import JobQueue


class SnapshotRestoreMetrics(unittest.TestCase):
    def test_roundtrip_and_detachment(self):
        queue = JobQueue(max_attempts=4)
        queue.submit("a", {"tags": [1]}, 2)
        queue.submit("b", {}, 1)
        queue.claim()
        snap = queue.snapshot()
        json.dumps(snap)
        before = copy.deepcopy(snap)
        restored = JobQueue.from_snapshot(snap)
        self.assertEqual(snap, before)
        self.assertEqual(restored.status("a").state, "pending")
        self.assertEqual(restored.status("a").attempts, 1)
        self.assertEqual(restored.claim().job_id, "a")
        restored.status("a").payload["tags"].append(2)
        self.assertEqual(queue.status("a").payload["tags"], [1])
        self.assertEqual(snap["jobs"][0]["payload"]["tags"], [1])

    def test_pending_order_and_retry_state(self):
        queue = JobQueue(max_attempts=3)
        queue.submit("a", {}, 5)
        queue.submit("b", {}, 5)
        queue.submit("c", {}, 5)
        queue.claim()
        queue.fail("a")
        snap = queue.snapshot()
        self.assertEqual([JobQueue.from_snapshot(snap).claim().job_id], ["b"])
        restored = JobQueue.from_snapshot(snap)
        self.assertEqual([restored.claim().job_id for _ in range(3)],
                         ["b", "c", "a"])

    def test_invalid_restore_atomic(self):
        api = QueueAPI()
        api.handle({"action": "submit", "id": "live"})
        good = api.handle({"action": "snapshot"})
        broken = copy.deepcopy(good)
        broken["pending"].append("missing")
        with self.assertRaises((ValueError, KeyError, TypeError)):
            api.handle({"action": "restore", "snapshot": broken})
        self.assertEqual(api.handle({"action": "status", "id": "live"})["state"],
                         "pending")
        api.handle({"action": "restore", "snapshot": good})
        self.assertEqual(api.handle({"action": "claim"})["id"], "live")

    def test_metrics_and_duplicate_rejection(self):
        api = QueueAPI()
        api.handle({"action": "submit", "id": "a"})
        api.handle({"action": "submit", "id": "b"})
        api.handle({"action": "claim"})
        api.handle({"action": "ack", "id": "a"})
        metrics = api.handle({"action": "metrics"})
        self.assertEqual(metrics["pending"], 1)
        self.assertEqual(metrics["done"], 1)
        self.assertEqual(metrics["total_attempts"], 1)
        snap = api.handle({"action": "snapshot"})
        snap["jobs"].append(copy.deepcopy(snap["jobs"][0]))
        with self.assertRaises((ValueError, KeyError, TypeError)):
            JobQueue.from_snapshot(snap)


if __name__ == "__main__":
    unittest.main()
