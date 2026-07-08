# TXDA normalization operator implementations.
#
# RMSNorm:      uses torch.nn.functional.rms_norm (pure torch ATen op
#               dispatched through torch_txda)
#               For residual case, uses torch-native add + rms_norm.
#
# GemmaRMSNorm: sgl_kernel.gemma_rmsnorm (no residual)
#               sgl_kernel.gemma_fused_add_rmsnorm (residual)
#               gemma_rmsnorm internally applies norm(x) * (1 + weight);
#               pass obj.weight.data (the raw trained weight, not +1 shifted).

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
    from sgl_kernel import gemma_fused_add_rmsnorm, gemma_rmsnorm
    if residual is not None:
        gemma_fused_add_rmsnorm(x, residual, obj.weight.data, obj.variance_epsilon)
        return x, residual
    return gemma_rmsnorm(x, obj.weight.data, obj.variance_epsilon)