import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'xpu_cpu_offload', Path(__file__).resolve().parents[2]/'runtime/xpu_cpu_offload.py')
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {
    'vllm.platforms': SimpleNamespace(current_platform=None),
    'vllm.logger': SimpleNamespace(init_logger=None),
    'vllm.v1.kv_offload.cpu.spec': SimpleNamespace(CPUOffloadingSpec=object),
}):
    spec.loader.exec_module(module)


class Descriptors(list):
    def tolist(self):
        return list(self)


class BoundedStagingTests(unittest.TestCase):
    def test_full_history_staging_is_bounded_by_bytes_not_descriptor_count(self):
        sizes = [3276800]*2800 + [3268608]*48
        ranges = list(module.bounded_copy_ranges(sizes))
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], len(sizes))
        self.assertTrue(all(sum(sizes[a:b]) <= 128*1024**2 for a,b in ranges))
        self.assertTrue(all(left[1] == right[0] for left,right in zip(ranges,ranges[1:])))

    def test_dma_finishes_before_another_staging_allocation_is_submitted(self):
        events=[]
        sizes=Descriptors([module.MAX_LOAD_BATCH_BYTES, 1, 0, 2])
        def copy(src,dst,part,**kwargs):
            events.append(('copy',src,dst,part,kwargs))
        module.bounded_load(copy,lambda:events.append('drained'),
                            [10,11,12,13],[20,21,22,23],sizes,
                            is_src_access_order_any=True)
        self.assertEqual([event if isinstance(event,str) else event[0] for event in events],
                         ['copy','drained','copy','drained'])
        self.assertEqual(events[0][1:4],([10],[20],[module.MAX_LOAD_BATCH_BYTES]))
        self.assertEqual(events[2][1:4],([11,12,13],[21,22,23],[1,0,2]))
        self.assertEqual(events[2][4],{'is_src_access_order_any':True})

    def test_invalid_descriptor_fails_before_it_can_allocate_oversized_staging(self):
        for size in [-1,module.MAX_LOAD_BATCH_BYTES+1]:
            with self.assertRaises(ValueError):
                list(module.bounded_copy_ranges([size]))

    def test_zero_payload_has_no_dma_or_staging(self):
        self.assertEqual(list(module.bounded_copy_ranges([0,0])),[])


if __name__=='__main__':unittest.main()
