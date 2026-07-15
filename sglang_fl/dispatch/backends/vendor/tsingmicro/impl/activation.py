# TXDA activation operator implementations.
#
# All implementations here use pure PyTorch operations.  torch_txda
# transparently maps these to TXDA hardware.
#
# silu_and_mul:  SiLU(gate) * up, the fused activation in SwiGLU FFN blocks.
#                Splits input [..., 2*d] into gate/up halves along the last dim.

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul_txda(obj, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return F.silu(x1) * x2
