# TXDA normalization operator implementations.
#
# IMPORTANT: Do NOT use sgl_kernel in this module.  On TXDA, sgl_kernel is
# stubbed out (see patches/platform_stubs.py) because it depends on CUDA
# compiled shared libraries.  Any call to sgl_kernel functions will silently
# return _StubObj instead of real tensors, causing hard-to-debug errors
# downstream (e.g. "cannot unpack non-iterable _StubObj object").
#
# All implementations here use pure PyTorch operations.  torch_txda
# transparently maps these to TXDA hardware.
#
# RMSNorm:        uses torch.nn.functional.rms_norm (pure torch ATen op
#                 dispatched through torch_txda)
#                 For residual case, uses torch-native add + rms_norm.
#
# GemmaRMSNorm: uses manual RMS computation + weight scaling.
#               gemma_rmsnorm(x) = rms_norm(x) * (1 + weight)

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_txda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None:
        x.add_(residual)
        residual.copy_(x)
    out = torch.nn.functional.rms_norm(
        x, (x.shape[-1],), obj.weight.data, obj.variance_epsilon
    )
    if residual is not None:
        return out, residual
    return out


def gemma_rms_norm_txda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Gemma-style RMS normalization using pure PyTorch.

    Difference from standard RMSNorm: weight semantics are (weight + 1.0).
    Uses the same in-place residual pattern as rms_norm_txda.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None:
        x.add_(residual)
        residual.copy_(x)

    orig_dtype = x.dtype
    x_fp = x.float()
    variance = x_fp.pow(2).mean(-1, keepdim=True)
    x_fp = x_fp * torch.rsqrt(variance + obj.variance_epsilon)
    out = (x_fp * (1.0 + obj.weight.float())).to(orig_dtype)

    if residual is not None:
        return out, residual
    return out