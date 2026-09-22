"""Evaluate actual scalar address expressions against independent tensor indexing.

This covers duplicated schedulers, not C++ compilation, DMA, or NPU execution.
"""
import itertools
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


class KernelAddresses(unittest.TestCase):
    def test_fwd_h_scheduler_copies(self):
        paths = list((ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd").rglob("block_scheduler_gdn_fwd_h*.hpp"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            src = re.search(r"offset.hSrcOffset = ([^;]+);", text)[1]
            dst = re.search(r"offset.hDstOffset = ([^;]+);", text)[1]
            for heads in (2, 3):
                data = torch.arange(2 * 3 * heads * 4 * 5).reshape(2, 3, heads, 4, 5)
                for b, c, h in itertools.product(range(2), range(2), range(heads)):
                    ns = dict(stream=SimpleNamespace(shapeBatchIdx=b, chunkOffset=0, chunkIdx=c, vHeadIdx=h),
                              totalChunks=3, vNumHead=heads, kHeadDim=4, vHeadDim=5, vBlockOffset=0)
                    source = eval(src, {"__builtins__": {}}, ns)
                    ns["offset"] = SimpleNamespace(hSrcOffset=source)
                    target = eval(dst, {"__builtins__": {}}, ns)
                    with self.subTest(path=str(path.relative_to(ROOT)), b=b, c=c, h=h):
                        self.assertEqual(data.flatten()[source], data[b, c, h, 0, 0])
                        self.assertEqual(data.flatten()[target], data[b, c + 1, h, 0, 0])
                # Packed sequence 1 starts at global chunk 2.
                ns["stream"] = SimpleNamespace(shapeBatchIdx=0, chunkOffset=2, chunkIdx=0, vHeadIdx=1)
                self.assertEqual(eval(src, {"__builtins__": {}}, ns), data[0, 2, 1, 0, 0])

    def test_kda_fused_h_copies(self):
        paths = list((ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel").rglob("chunk_kda_fwd_*.h"))
        checked = 0
        for path in paths:
            source = path.read_text(encoding="utf-8")
            match = re.search(r"HOffset\([^)]*uint64_t r\) const\s*\{\s*return ([^;]+);", source)
            if match is None: continue
            checked += 1
            for heads in (2, 3):
                data = torch.arange(2 * 3 * heads * 4 * 5).reshape(2, 3, heads, 4, 5)
                for b, c, h, d, r in itertools.product(range(2), range(3), range(heads), range(4), range(5)):
                    offset = eval(match[1], {"__builtins__": {}}, dict(b=b, hv=h, chunkIdx=c,
                                  d=d, r=r, HV_=heads, NT_=3, K_=4, V_=5))
                    self.assertEqual(data.flatten()[offset], data[b, c, h, d, r])
        self.assertEqual(checked, 6)


if __name__ == "__main__":
    unittest.main()
