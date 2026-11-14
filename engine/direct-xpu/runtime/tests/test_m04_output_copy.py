"""Exercise the actual run() output path with a deterministic stand-in kernel."""
import unittest
from unittest.mock import patch
import torch
from exl3xpu import shared_kv_verify as m04


class CopyContract(unittest.TestCase):
    def test_native_output_and_no_output_preserve_all_half_bitpatterns(self):
        for batch in (1, 2, 4):
            for rows in (2, 3, 4, 5):
                with self.subTest(batch=batch, rows=rows):
                    count=batch*rows*24*256
                    bits=((torch.arange(count,dtype=torch.int64)%65536)-32768).to(torch.int16)
                    y=bits.view(torch.float16).reshape(batch,rows*24,256)
                    expected=m04.unpacked_output(y,batch,rows)
                    def kernel(*args):
                        args[8].copy_(y)
                    for has_out in (True, False):
                        q=torch.zeros((batch*rows,24,256),dtype=torch.float16)
                        out=torch.empty_like(q) if has_out else None
                        d=dict(q=q,k=q,v=q,block_table=torch.zeros((batch,1),dtype=torch.int32),
                               cu_seqlens_q=torch.arange(batch+1,dtype=torch.int32),
                               seqused_k=torch.ones(batch,dtype=torch.int32),
                               k_descale=torch.ones(1),v_descale=torch.ones(1),
                               max_seqlen_q=rows,max_seqlen_k=4096,out=out)
                        with patch.object(torch.ops.b70_exl3_attention,'shared_kv_verify_out',kernel,create=True):
                            actual=m04.run(d)
                        self.assertTrue(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))
                        if has_out:
                            self.assertIs(actual,out)
                            self.assertEqual(actual.data_ptr(),out.data_ptr())


if __name__=='__main__':unittest.main()
