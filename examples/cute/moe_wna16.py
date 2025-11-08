"""
This module implements a Mixture of Experts (MoE) layer with weight-only quantization 
using the Hexcute DSL. The implementation includes:

1. A quantized linear layer for MoE (MoELinearWnA16)
2. A fused MoE layer that combines two linear layers with activation (FusedMoEWnA16)
3. Various helper functions for memory layout and tensor operations

The code uses a high-level DSL called Hexcute to generate efficient CUDA code for 
NVIDIA GPUs. The implementation supports:
- Weight-only quantization (4-bit)
- Dynamic expert routing
- Fused operations for better performance
- Configurable block sizes and pipeline stages
- Asynchronous MMA pipeline for H100 GPUs

Key features of the H100 implementation:
1. Uses WGMMA (Warp Group Matrix Multiply-Accumulate) instructions
2. Implements a multi-stage pipeline for asynchronous computation
3. Optimizes memory access patterns for coalesced loads
4. Handles expert routing and token alignment efficiently
"""
import functools
import itertools
from typing import Any, Callable, Optional, Union

import hidet
import torch
from hidet.ffi import runtime_api
from hidet.graph.frontend.torch.utils import dtype_to_torch
from hidet.ir.cute import (ComposedTensorLayout, TensorLayout, auto_layout,
                           layout_auto, make_layout)
from hidet.ir.cute.algorithm import MmaAtom, TiledMma, auto_copy
from hidet.ir.cute.layout import (Level, canonicalize_thread_value_layout,
                                  composition, logical_divide)
from hidet.ir.cute.ops import (cast, copy, fill, make_tensor, mask, mma,
                               partition_dst, partition_src, rearrange,
                               tensor_view)
from hidet.ir.dtypes import f16, f32, i32, i64, u4
from hidet.ir.expr import Expr, symbol_var
from hidet.ir.library import tune
from hidet.ir.primitives.cuda.wgmma import (wgmma_commit_group, wgmma_fence,
                                            wgmma_wait_group)
from hidet.ir.type import DataType, data_type
from hidet.lang import attrs, grid
from hidet.lang.cuda import (blockIdx, cp_async_commit_group,
                             cp_async_wait_group, syncthreads)
from hidet.utils import initialize
from hidet.utils.benchmark import do_bench
from hidet.utils.py import cdiv

from moe_utils import (moe_align_block_size_kernel, moe_sum_kernel,
                              silu_and_mul_kernel)
from hidet.lang.cuda import threadIdx
from hidet.ir.primitives.cuda.mutex import acquire_seq_semaphore, release_seq_semaphore
from hidet.ir.primitives.cuda.atomic import atomic_add
from hidet.ir.cute.contexts import warp_groups_consumer, warp_groups_producer


class Function:
    """A wrapper class for functions that provides a consistent interface."""

    def __init__(self, raw_function):
        self.raw_function = raw_function

    def __call__(self, *args, **kwargs):
        return self.raw_function(*args, **kwargs)

    def __repr__(self):
        return self.raw_function.__name__


@Function
def a_addr_function(addr: Expr, expert_lut: Expr, experts_per_token: int,
                    features: int):
    """Compute the address for accessing input tensor A based on expert routing.

    Args:
        addr: The base address
        expert_lut: Lookup table for expert indices
        experts_per_token: Number of experts per token
        features: Number of features per token

    Returns:
        The computed address for accessing input tensor A
    """
    m_crd = addr // features
    return addr - m_crd * features + (expert_lut[m_crd] //
                                    experts_per_token) * features


@Function
def c_addr_function(addr: Expr, expert_lut: Expr, features: int):
    """Compute the address for accessing output tensor C based on
    expert routing.

    Args:
        addr: The base address
        expert_lut: Lookup table for expert indices
        features: Number of features per token

    Returns:
        The computed address for accessing output tensor C
    """
    m_crd = addr // features
    return addr - m_crd * features + expert_lut[m_crd] * features


@Function
def rw_addr_function(addr: Expr, expert_lut: Expr, features: int):
    """Compute the address for accessing routing weights based on
    expert routing.

    Args:
        addr: The base address
        expert_lut: Lookup table for expert indices
        features: Number of features per token

    Returns:
        The computed address for accessing routing weights
    """
    m_crd = addr // features
    return addr - m_crd * features + expert_lut[m_crd]


class Config:
    pass


class MoEConfig(Config):
    """Configuration class for MoE layer parameters.

    This class defines the configuration parameters for the MoE layer,
    including:
    - Tiled matrix multiplication settings
    - Block sizes
    - Pipeline stages
    - Parallel processing settings
    """

    def __init__(self,
                tiled_mma: TiledMma,
                block_k: int,
                stages: int,
                parallel_k_slices: int = 1):
        """Initialize the MoE configuration.

        Args:
            tiled_mma: Tiled matrix multiplication configuration
            block_k: Block size in the K dimension
            stages: Number of pipeline stages
            parallel_k_slices: Number of parallel K dimension slices (default: 1)

        The initialization:
        1. Stores the tiled MMA configuration
        2. Sets the block size in K dimension
        3. Extracts block sizes from tiled MMA layouts
        4. Sets up parallel processing parameters
        """
        self.tiled_mma = tiled_mma
        self.block_k = block_k
        a_shape, _ = self.tiled_mma.a_tv_layout()
        b_shape, _ = self.tiled_mma.b_tv_layout()

        block_m, k_tile = a_shape
        block_n, _ = b_shape
        self.block_m = block_m
        self.block_n = block_n
        self.parallel_k_slices = parallel_k_slices
        self.k_tile = k_tile
        self._stages = stages

    def __str__(self):
        """Generate a string representation of the configuration.

        Returns:
            A formatted string containing all configuration parameters:
            - tiled_mma configuration
            - block_k size
            - number of stages
            - parallel_k_parts setting
        """
        indent = " " * 2
        return ("{\n" +
                f"{indent}tiled_mma: {self.tiled_mma.str_indented(2)},\n" +
                f"{indent}block_k: {self.block_k},\n" +
                f"{indent}stages: {self.stages},\n" +
                f"{indent}parallel_k_parts: {self.parallel_k_parts},\n" + "}")

    @property
    def threads(self):
        """Get the number of threads per block.

        Returns:
            The total number of threads calculated from the C tensor layout.
            This is determined by:
            1. Getting the C tensor value layout
            2. Canonicalizing the thread value layout
            3. Computing the total size of the thread dimension
        """
        _, c_tv_layout = self.tiled_mma.c_tv_layout()
        c_t, _ = canonicalize_thread_value_layout(c_tv_layout)
        return c_t.size()

    # These properties return the number of elements in tensor A, B, C
    # per thread. They are currently not used in our current kernel because the compiler
    # will handle the allocation of register tensors. They are kept here because
    # they performance model will query these properties to estimate the register
    # count.
    @property
    def a_elements(self):
        """Get the number of elements in tensor A per thread.

        Returns:
            The number of elements in tensor A that each thread processes.
            This is calculated by:
            1. Getting the A tensor value layout
            2. Canonicalizing the thread value layout
            3. Computing the size of the value dimension
        """
        _, a_tv_layout = self.tiled_mma.a_tv_layout()
        _, a_v = canonicalize_thread_value_layout(a_tv_layout)
        return a_v.size()

    @property
    def b_elements(self):
        """Get the number of elements in tensor B per thread.

        Returns:
            The number of elements in tensor B that each thread processes.
            This is calculated by:
            1. Getting the B tensor value layout
            2. Canonicalizing the thread value layout
            3. Computing the size of the value dimension
        """
        _, b_tv_layout = self.tiled_mma.b_tv_layout()
        _, b_v = canonicalize_thread_value_layout(b_tv_layout)
        return b_v.size()

    @property
    def c_elements(self):
        """Get the number of elements in tensor C per thread.

        Returns:
            The number of elements in tensor C that each thread processes.
            This is calculated by:
            1. Getting the C tensor value layout
            2. Canonicalizing the thread value layout
            3. Computing the size of the value dimension
        """
        _, c_tv_layout = self.tiled_mma.c_tv_layout()
        _, c_v = canonicalize_thread_value_layout(c_tv_layout)
        return c_v.size()

    # This is not used in our current kernel but kept here for performance model.
    def scale_elements(self, group_size: int = 64):
        """Calculate the number of scale elements per thread.

        Args:
            group_size: Size of quantization groups (default: 64)

        Returns:
            The number of scale elements per thread, which depends on:
            1. The block size in N dimension
            2. The block size in K dimension
            3. The group size for quantization
            4. The layout of tensor B

        The calculation handles two cases:
        - When block_k > group_size: Uses a composed layout with group_size
        - When block_k <= group_size: Uses a simple layout
        """
        _, bn, bk = self.thread_block_shape
        _, b_tv_layout = self.tiled_mma.b_tv_layout()
        _, b_v = canonicalize_thread_value_layout(b_tv_layout)
        if bk > group_size:
            scale_v = composition(
                TensorLayout((bn, (group_size, bk // group_size)),
                            (1, (0, bn * group_size))),
                b_v,
            )
        else:
            scale_v = composition(TensorLayout((bn, bk), (1, 0)), b_v)
        return scale_v.count()

    # This is not used in our current kernel but kept here for performance model.
    def bias_elements(self, group_size):
        """Calculate the number of bias elements per thread.

        Args:
            group_size: Size of quantization groups

        Returns:
            The number of bias elements per thread, which is identical to
            the number of scale elements since they share the same layout.
        """
        return self.scale_elements(group_size)

    @property
    def parallel_k_parts(self):
        """Get the number of parallel K dimension parts.

        Returns:
            The number of parallel slices in the K dimension,
            which is equal to parallel_k_slices.
        """
        return self.parallel_k_slices

    @property
    def thread_block_shape(self):
        """Get the shape of the thread block.

        Returns:
            A tuple containing (block_m, block_n, block_k) representing:
            - block_m: Block size in M dimension
            - block_n: Block size in N dimension
            - block_k: Block size in K dimension
        """
        return self.block_m, self.block_n, self.block_k

    @property
    def stages(self):
        """Get the number of pipeline stages.

        Returns:
            The number of stages in the pipeline, which determines
            the level of instruction overlap and memory latency hiding.
        """
        return self._stages

    # These two functions are used to estimate the shared memory usage
    # for the MoE layer. They help us to reject the invalid config with
    # too large shared memory usage. The actual shared memory usage will
    # be calculated in a compiler pass.
    def dynamic_smem_bytes_per_stage(self,
                                    a_dtype: DataType,
                                    b_dtype: DataType = u4,
                                    group_size: int = 64):
        """Calculate the shared memory usage per pipeline stage.

        Args:
            a_dtype: Data type for tensor A
            b_dtype: Data type for tensor B (default: u4)
            group_size: Size of quantization groups (default: 64)

        Returns:
            The total shared memory bytes per stage, which includes:
            1. Memory for tensor A: block_m * block_k * a_dtype.nbytes
            2. Memory for tensor B: block_n * block_k * b_dtype.nbits / 8
            3. Memory for scale: block_n * ceil(block_k/group_size) * a_dtype.nbytes
            4. Memory for bias: block_n * ceil(block_k/group_size) * a_dtype.nbytes
        """
        smem_a = self.block_m * self.block_k * a_dtype.nbytes
        smem_b = self.block_n * self.block_k * b_dtype.nbits // 8
        smem_scale = self.block_n * cdiv(self.block_k,
                                        group_size) * a_dtype.nbytes
        smem_bias = self.block_n * cdiv(self.block_k,
                                        group_size) * a_dtype.nbytes
        return smem_a + smem_b + smem_scale + smem_bias

    def dynamic_smem_bytes(self,
                        a_dtype: DataType,
                        b_dtype: DataType = u4,
                        group_size: int = 64,
                        stages=1):
        """Calculate the total shared memory usage.

        Args:
            a_dtype: Data type for tensor A
            b_dtype: Data type for tensor B (default: u4)
            group_size: Size of quantization groups (default: 64)
            stages: Number of pipeline stages (default: 1)

        Returns:
            The total shared memory bytes required, which is:
            dynamic_smem_bytes_per_stage * stages
        """
        dyn_smem_bytes_per_stage = self.dynamic_smem_bytes_per_stage(
            a_dtype, b_dtype, group_size)
        return dyn_smem_bytes_per_stage * stages


_predefined_config: list[MoEConfig] = []


@initialize()
def register_configs():
    # This is the predefined config for the MoE layer. They are same as the
    # configs in other quantization kernels like cserve_awq_hidet_kernel.py and
    # cserve_gptq_hidet_kernel.py. Please refer to these files for more details.
    PARALLEL_K_PARTS = [1, 8]
    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4),
                                TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(
            MoEConfig(tiled_mma, 128, 4, parallel_k_parts))
    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8),
                                TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(
            MoEConfig(tiled_mma, 64, 8, parallel_k_parts))
        _predefined_config.append(
            MoEConfig(tiled_mma, 128, 4, parallel_k_parts))

    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (2, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4),
                                TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(
            MoEConfig(tiled_mma, 128, 4, parallel_k_parts))
    
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4),
                                TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(
            MoEConfig(tiled_mma, 128, 2, parallel_k_parts))

    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (6, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4),
                                TensorLayout((1, 4)), (1, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(MoEConfig(tiled_mma, 32, 4,
                                            parallel_k_parts))
        _predefined_config.append(MoEConfig(tiled_mma, 64, 4,
                                            parallel_k_parts))

    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (8, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4),
                                TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(MoEConfig(tiled_mma, 32, 4,
                                            parallel_k_parts))
        _predefined_config.append(MoEConfig(tiled_mma, 64, 4,
                                            parallel_k_parts))

    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (8, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4),
                                TensorLayout((1, 4)), (1, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(MoEConfig(tiled_mma, 32, 4,
                                            parallel_k_parts))
        _predefined_config.append(MoEConfig(tiled_mma, 64, 4,
                                            parallel_k_parts))

    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (8, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4),
                                TensorLayout((1, 4)), (1, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(MoEConfig(tiled_mma, 32, 4,
                                            parallel_k_parts))
        _predefined_config.append(MoEConfig(tiled_mma, 64, 4,
                                            parallel_k_parts))

    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (8, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8),
                                TensorLayout((1, 8)), (2, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(MoEConfig(tiled_mma, 64, 4,
                                            parallel_k_parts))

    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (8, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8),
                                TensorLayout((1, 8)), (2, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(MoEConfig(tiled_mma, 64, 4,
                                            parallel_k_parts))

    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (8, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8),
                                TensorLayout((1, 8)), (4, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(MoEConfig(tiled_mma, 64, 4,
                                            parallel_k_parts))


_predefined_hopper_config: list[MoEConfig] = []


@initialize()
def register_configs_hopper():
    # This are the predefined configs for the MoE layer on Hopper architecture.
    # Currently, the kernels are not used in the MoE layer because Hidet doesn't
    # support WGMMA instructions yet. After Hidet's support for WGMMA instructions is
    # ready, the kernels can be used in the MoE layer.
    # Note: currently, the Hopper kernels don't help much because of the memory-bound
    # nature of the MoE layer. Need to further investigate if warp specialization
    # can help.
    for n in [8, 16, 32, 48, 64, 96, 128, 192, 256]:
        a = TensorLayout(((128, ), (n, 16)), ((0, ), (1, n)))
        b = TensorLayout(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512)))
        c = TensorLayout(((4, 8, 4), (2, 2, n // 8)),
                        ((2, n, 16 * n), (1, 8 * n, 8)))
        bk = 128 if n < 64 else 64
        mma_atom = MmaAtom("warp_group", (n, 64, 16), a, b, c, c, (1, 2))
        wg_in_threadblock = Level("warp_group", "thread_block", (1, 1),
                                TensorLayout((1, 1)), (1, 1))
        tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])
        _predefined_hopper_config.append(MoEConfig(tiled_mma, bk, 4, 1))

        mma_atom = MmaAtom("warp_group", (n, 64, 16), a, b, c, c, (1, 1))
        wg_in_threadblock = Level("warp_group", "thread_block", (1, 2),
                                TensorLayout((1, 2)), (1, 1))
        tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])
        _predefined_hopper_config.append(MoEConfig(tiled_mma, bk, 4, 1))



def create_fake_tensors(
    num_tokens: int,
    experts_per_token: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    group_size: int,
    param_dtype: Union[str, DataType],
    act_dtype: Union[str, DataType] = "float16",
    return_topk: Optional[bool] = True,
):
    lo = -3
    hi = 3
    device = "cuda"
    bdtype = torch.int32
    factor = bdtype.itemsize * 8 // param_dtype.nbits
    adtype = dtype_to_torch(act_dtype)
    scale_dtype = dtype_to_torch(act_dtype)
    bias_dtype = dtype_to_torch(act_dtype)
    a = (torch.randint(low=lo,
                    high=hi,
                    size=(num_tokens, in_features),
                    dtype=adtype,
                    device=device) / in_features)
    b = torch.randint(
        low=0,
        high=hi,
        size=(num_experts, in_features, out_features // factor),
        dtype=bdtype,
        device=device,
    )
    scale = torch.randint(
        low=-1,
        high=2,
        size=(num_experts, in_features // group_size, out_features),
        dtype=scale_dtype,
        device=device,
    )
    zeros = torch.randint(
        low=0,
        high=hi,
        size=(num_experts, in_features // group_size, out_features),
        dtype=bias_dtype,
        device=device,
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, experts_per_token),
        dtype=torch.uint32,
        device=device,
    )
    topk_weights = torch.randint(
        low=-2,
        high=2,
        size=(num_tokens, experts_per_token),
        dtype=torch.float32,
        device=device,
    )
    if return_topk:
        return a, b, scale, zeros, topk_weights, topk_ids
    else:
        return a, b, scale, zeros


#basic_block = TensorLayout(((8, 2), (2, 4, 2)), ((32, 4), (1, 8, 2)))
basic_block = TensorLayout(((8, 2), (2, 4, 2)), ((4, 2), (1, 64, 32)))

def deduce_gmem_layout(k: int, block_k: int, block_n: int, stages: int = 1):
    # magic gmem and smem layout that coalesce global memory load and
    # resolve shared memory bank conflict
    # This layout is inferred using shared memory constraints of instructions
    # like ldmatrix, cp_async, etc.
    m_mode, n_mode = basic_block
    n_shape = n_mode.shape + (block_k // n_mode.size(), )
    n_stride = n_mode.stride + (basic_block.cosize(), )
    n_mode_ = TensorLayout(n_shape, n_stride)
    m_shape = m_mode.shape + (block_n // m_mode.size(), )
    cosize = block_k // 16 * basic_block.cosize()
    m_stride = m_mode.stride + (cosize, )
    m_mode_ = TensorLayout(m_shape, m_stride)
    if stages > 1:
        smem_layout = make_layout(m_mode_, n_mode_)
        stage_layout = TensorLayout(stages, smem_layout.cosize())
        smem_layout = make_layout(m_mode_, n_mode_, stage_layout)
    else:
        smem_layout = make_layout(m_mode_, n_mode_)

    n_shape = n_mode.shape + (k // n_mode.size(), )
    n_stride = n_mode.stride + (basic_block.cosize(), )
    n_mode_ = TensorLayout(n_shape, n_stride)
    m_shape = m_mode.shape + (block_n // m_mode.size(), )
    cosize = k // 16 * basic_block.cosize()
    m_stride = m_mode.stride + (cosize, )
    m_mode_ = TensorLayout(m_shape, m_stride)
    gmem_layout = make_layout(m_mode_, n_mode_)
    return gmem_layout, smem_layout


def preprocess_weight(weight: torch.Tensor):
    """
    AWQ define a interleaved format for quantized parameters such as
    weights, scales, and zero points. This function is used to convert
    the interleaved quantized parameters to the optimal format used in Hidet.
    """
    e, m, n = weight.shape
    dtype = weight.dtype
    element_size = weight.element_size()
    pack_factor = element_size * 8 // u4.nbits
    n = n * pack_factor
    w = torch.empty(e, m, n // pack_factor, dtype=dtype, device="cuda")
    bm, bn = 64, 64
    threads = 128
    assert m % bm == 0 and n % bn == 0

    if not weight.is_contiguous():
        weight = weight.contiguous()
    m_mode, n_mode = basic_block
    n_shape = n_mode.shape + (m // n_mode.size(), )
    n_stride = n_mode.stride + (basic_block.cosize(), )
    n_mode = TensorLayout(n_shape, n_stride)
    m_shape = m_mode.shape + (n // m_mode.size(), )
    cosize = m // 16 * basic_block.cosize()
    m_stride = m_mode.stride + (cosize, )
    m_mode = TensorLayout(m_shape, m_stride)
    gmem_layout = make_layout(n_mode, m_mode)

    layout = TensorLayout((m, n))
    tile = TensorLayout((bm, bn), (1, m))
    tile = logical_divide(layout, tile)
    tile = composition(gmem_layout, tile)
    gmem, strides = tile
    m_stride, n_stride = strides.stride
    m_stride //= bm
    n_stride //= bn * m

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: u4[e, m, n], wo: u4[e, n, m]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), e
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            bidy = i64(blockIdx.y)
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_wi = tensor_view(
                wi[bidy, pid_m * bm:, pid_n * bn:],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )
            tr_wi = make_tensor(u4, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi)

            tr_w = cast(tr_wi, f16)
            tr_w_cvt = rearrange(tr_w, auto_layout, "register")
            tr_wo = cast(tr_w_cvt, u4)

            tg_wo = tensor_view(
                wo[bidy, pid_n * bn * n_stride:, pid_m * bm * m_stride:],
                gmem,
                "global",
            )
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo)

    func = script_module.build()
    func(weight, w)
    return w


def compute_max_num_tokens_padded_and_max_num_m_blocks(num_tokens: int,
                                                    experts_per_token: int,
                                                    num_experts: int,
                                                    block_size: int):
    max_num_tokens_padded = num_tokens * experts_per_token + num_experts * (
        block_size - 1)
    max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)
    return max_num_tokens_padded, max_num_m_blocks


class MoELinearWnA16:
    """A quantized linear layer for Mixture of Experts (MoE) with weight-only
    quantization.

    This class implements a linear layer for MoE that uses 4-bit weight
    quantization. It supports:
    - Dynamic expert routing
    - Weight-only quantization (4-bit)
    - Configurable block sizes and pipeline stages
    - Efficient memory access patterns

    Attributes:
        M_BINS: List of supportiiied block sizes for the M dimension
    """

    M_BINS = [8, 16, 24, 32, 40, 48, 64, 128, 2048, 4096]

    def __init__(
        self,
        experts_per_token: int,
        in_features: int,
        out_features: int,
        num_experts: int,
        group_size: int,
        param_dtype: Union[str, DataType],
        act_dtype: Union[str, DataType] = "float16",
        divide_by_experts_per_token: bool = True,
        mul_routed_weight: bool = False,
        # These two arguments are used in the ablation study
        triton_dataflow: bool = False,
        triton_shared_memory: bool = False,
    ):
        """Initialize the MoE linear layer.

        Args:
            experts_per_token: Number of experts per token
            in_features: Number of input features
            out_features: Number of output features
            num_experts: Total number of experts
            group_size: Size of quantization groups
            param_dtype: Data type for parameters (weights)
            act_dtype: Data type for activations
            divide_by_experts_per_token: Whether to divide input by experts_per_token
            mul_routed_weight: Whether to multiply by routing weights
        """
        self.experts_per_token = experts_per_token
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.group_size = group_size
        self.param_dtype = data_type(param_dtype)
        self.act_dtype = data_type(act_dtype)
        self.divide_by_experts_per_token = divide_by_experts_per_token
        self.mul_routed_weight = mul_routed_weight
        self.cache = {}
        self.triton_dataflow = triton_dataflow
        self.triton_shared_memory = triton_shared_memory

    def get_config(self, num_tokens):
        """Get the optimal configuration for the given number of tokens.

        Args:
            num_tokens: Number of input tokens

        Returns:
            The optimal configuration for the given number of tokens
        """
        m_clip = min(max(num_tokens, 8), 4096)
        m_roundup = min(i for i in self.M_BINS if i >= m_clip)
        assert m_roundup in self.M_BINS
        return self.cache[m_roundup]

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        scale: torch.Tensor,
        zeros: torch.Tensor,
        sorted_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        total_tokens_per_expert: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
        expert_start_index: torch.Tensor,
        routed_weight: Optional[torch.Tensor],
    ):
        """Execute the MoE linear layer computation.

        Args:
            a: [num_tokens, in_features] Input hidden states
            b: [num_experts, out_features, in_features] Quantized expert weights
            c: [num_tokens * experts_per_token, out_features] Output states
            scale: [num_experts, in_features//group_size, out_features] Scale
            zeros: [num_experts, in_features//group_size, out_features] Zero point
            sorted_ids: [max_num_tokens_padded] Sorted expert indices
            expert_ids: [max_num_m_blocks] Block ID to expert ID lookup table
            total_tokens_per_expert: [num_experts] Token count per expert
            expert_start_index: [num_experts] Starting block ID per expert
            routed_weight: [num_tokens, experts_per_token] Routing weights

        Example:
            For 4 tokens, 4 experts, and 2 experts per token, if topk_ids is:
            [[0, 1], [2, 3], [1, 2], [0, 3]]

            Then aggregated by expert ID:
            [[0, 7], [1, 4], [2, 5], [3, 6]]

            With block_m=8, padded sorted_topk_ids becomes:
            [0, 7, 0, 0, 0, 0, 0, 0,  # Expert 0
            1, 4, 0, 0, 0, 0, 0, 0,  # Expert 1
            2, 5, 0, 0, 0, 0, 0, 0,  # Expert 2
            3, 6, 0, 0, 0, 0, 0, 0]  # Expert 3

            This serves as a lookup table converting m coordinates to token indices.
            num_tokens_post_pad would be 32 in this case.
        """
        num_tokens, _ = a.shape
        if not self.divide_by_experts_per_token:
            num_tokens //= self.experts_per_token
        config, moe_linear = self.get_config(num_tokens)
        if config.parallel_k_parts > 1:
            parallel_k_parts = config.parallel_k_parts
            max_num_tokens_padded = num_tokens * self.experts_per_token + self.num_experts * (
                config.block_m - 1)
            max_num_m_blocks = cdiv(max_num_tokens_padded, config.block_m)
            grid_n = cdiv(self.out_features, config.block_n)
            c_partial = torch.empty(
                (parallel_k_parts, num_tokens * self.experts_per_token,
                 self.out_features),
                dtype=getattr(torch, self.act_dtype.name),
                device=a.device,
            )
            locks = torch.zeros((max_num_m_blocks, grid_n),
                                dtype=torch.int32,
                                device=a.device)
            moe_linear(a, b, c, scale, zeros, sorted_ids, expert_ids,
                       total_tokens_per_expert, expert_start_index,
                       num_tokens_post_pad, routed_weight, c_partial, locks)
        else:
            moe_linear(
                a,
                b,
                c,
                scale,
                zeros,
                sorted_ids,
                expert_ids,
                total_tokens_per_expert,
                expert_start_index,
                num_tokens_post_pad,
                routed_weight,
            )

    def unpack_parameters(self):
        """Unpack the layer parameters.

        Returns:
            Tuple of (experts_per_token, in_features, out_features, num_experts,
                    group_size, param_dtype, act_dtype)
        """
        return (
            self.experts_per_token,
            self.in_features,
            self.out_features,
            self.num_experts,
            self.group_size,
            self.param_dtype,
            self.act_dtype,
        )

    @tune.space(1, config=[_predefined_config[0]])
    @tune.space(2, config=_predefined_config + _predefined_hopper_config)
    def modules(self, config: MoEConfig):
        """Generate the IR modules for the MoE linear layer.

        Args:
            config: Configuration for the MoE layer

        Returns:
            The IR module for the MoE linear layer
        """
        if self.triton_dataflow:
            tune.check(config not in _predefined_hopper_config)
            tune.check(config.parallel_k_parts == 1)
            tune.check(config.block_k >= 64)
            return self._moe_wna16_triton_dataflow(config)
        if self.triton_shared_memory:
            tune.check(config not in _predefined_hopper_config)
            tune.check(config.parallel_k_parts == 1)
            return self._moe_wna16_bad_smem_layout(config)
        major, minor = hidet.cuda.compute_capability()
        if config in _predefined_hopper_config:
            tune.check(False)
            tune.check(major >= 9)
            return self._moe_wna16_hopper(config)
        else:
            tune.check(config.stages >= 2)
            if config.parallel_k_parts > 1:
                return self._moe_wna16_split_k(config)
            else:
                #return self._moe_wna16_triton_dataflow(config)
                #return self._moe_wna16_bad_smem_layout(config)
                return self._moe_wna16(config)

    # right now, this kernel is not used
    def _moe_wna16_hopper(self, config: MoEConfig):
        """
        Implementation of MoE linear layer optimized for H100 GPUs.
        Key features:
        1. Uses WGMMA instructions for matrix multiplication
        2. Implements a multi-stage pipeline for asynchronous computation
        3. Optimizes memory access patterns for coalesced loads
        4. Handles expert routing and token alignment efficiently

        Pipeline stages:
        1. Load data from global memory to shared memory
        2. Dequantize weights using scale and bias
        3. Perform matrix multiplication using WGMMA
        4. Write results back to global memory

        The pipeline uses:
        - mbarriers for synchronization between stages
        - Double buffering in shared memory
        - Asynchronous memory operations
        - WGMMA instructions for matrix multiplication
        """
        (
            experts_per_token,
            in_features,
            out_features,
            num_experts,
            group_size,
            param_dtype,
            act_dtype,
        ) = self.unpack_parameters()

        from hidet.ir.cute.ops import (make_mbarriers, mbarrier_arrive,
                                       mbarrier_wait, partition_A,
                                       wgmma_fence_operand)

        # Get configuration parameters for tiled matrix multiplication
        tiled_mma = config.tiled_mma
        bk = config.block_k  # Block size in K dimension
        k_pipe_max = config.stages  # Number of pipeline stages
        k_pipe_mmas = (k_pipe_max - 2 if k_pipe_max > 2 else k_pipe_max - 1
                    )  # Number of MMA stages in pipeline
        bm, bn, _ = config.thread_block_shape  # Block sizes in M and N dimensions
        threads = config.threads  # Number of threads per block
        k_tile = config.k_tile  # Tile size for K dimension

        # Define memory layouts optimized for H100
        # These layouts ensure coalesced memory access and avoid bank conflicts
        gmem_layout, smem_layout = deduce_gmem_layout(in_features, bk, bn,
                                                    k_pipe_max)

        # Define layouts for scale and bias tensors
        scale_gmem_layout = TensorLayout(
            (bn, (group_size, in_features // group_size)),
            (1, (0, out_features)))

        # Define shared memory layouts for scale and bias with pipeline stages
        if bk > group_size:
            scale_smem_layout = TensorLayout(
                (bn, (group_size, bk // group_size), k_pipe_max),
                (1, (0, bn), bn * bk // group_size),
            )
        else:
            scale_smem_layout = TensorLayout((bn, bk, k_pipe_max), (1, 0, bn))

        # Set up address calculation functions for MoE routing
        divisor = experts_per_token if self.divide_by_experts_per_token else 1
        functor_a = functools.partial(a_addr_function,
                                    experts_per_token=divisor,
                                    features=in_features)
        functor_c = functools.partial(c_addr_function, features=out_features)
        functor_rw = functools.partial(rw_addr_function, features=out_features)

        # Calculate dimensions for padded tensors and blocks
        num_tokens = symbol_var("num_tokens")
        max_num_tokens_padded = num_tokens * experts_per_token + num_experts * (
            bm - 1)
        max_num_m_blocks = cdiv(max_num_tokens_padded, bm)
        mul_routed_weight = self.mul_routed_weight
        tune.check(in_features %
                bk == 0)  # Ensure block size divides input features
        tma_copy_tx = (
            (bm * bk) * act_dtype.nbytes + (bn * bk) * param_dtype.nbits // 8 +
            bn * ((bk + group_size - 1) // group_size) * act_dtype.nbytes * 2)
        unroll = f"u{k_pipe_max}"

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                    a: act_dtype[num_tokens, in_features],  # Input tensor
                    b: param_dtype[num_experts, out_features,
                                in_features],  # Quantized weights
                    c: act_dtype[num_tokens * experts_per_token,
                                out_features],  # Output tensor
                    scale: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Scale factors
                    bias: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Zero points
                    sorted_topk_ids: i32[
                        max_num_tokens_padded],  # Sorted expert indices
                    expert_ids: i32[
                        max_num_m_blocks],  # Expert ID lookup table
                    total_tokens_per_expert: i32[
                        num_experts],  # Token count per expert
                    expert_start_index: i32[
                        num_experts],  # Starting index per expert
                    num_tokens_post_pad: ~i32,  # Number of tokens after padding
                    routed_weight: ~f32,  # Optional routing weights
            ):
                """
                H100-optimized MoE linear layer kernel.

                Pipeline stages:
                1. Load data from global memory to shared memory
                2. Dequantize weights using scale and bias
                3. Perform matrix multiplication using WGMMA
                4. Write results back to global memory

                The pipeline uses:
                - mbarriers for synchronization between stages
                - Double buffering in shared memory
                - Asynchronous memory operations
                - WGMMA instructions for matrix multiplication
                """
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(max_num_tokens_padded, bm) * cdiv(
                    out_features, bn)
                attrs.cuda.dynamic_smem_bytes = 0

                # Initialize starting positions for K dimension
                k_start_pos = 0
                k_start_ofs = gmem_layout((0, k_start_pos))

                # Calculate thread block and grid dimensions
                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(max_num_tokens_padded, bm)
                num_pid_n = cdiv(out_features, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                num_tokens_post_pad_value = num_tokens_post_pad[0]
                if pid_m * bm >= num_tokens_post_pad_value:
                    return
                pid_n = (pid % num_pid_in_group) // group_size_m

                expert_idx = i64(expert_ids[pid_m])
                expert_start_pid = expert_start_index[expert_idx]
                expert_total_tokens = total_tokens_per_expert[expert_idx]

                # Initialize mbarriers for pipeline synchronization
                mbar_mma = make_mbarriers(k_pipe_max)
                mbar_tma = make_mbarriers(k_pipe_max)

                # Allocate shared memory tensors for pipeline stages
                ts_a = make_tensor(act_dtype, layout_auto(
                    (bm, bk, k_pipe_max)), "shared")
                ts_b = make_tensor(param_dtype, smem_layout, "shared")
                ts_scale = make_tensor(act_dtype, scale_smem_layout, "shared")
                ts_bias = make_tensor(act_dtype, scale_smem_layout, "shared")

                # Allocate register tensors for computation
                tr_b = make_tensor(param_dtype, layout_auto((bn, k_tile * 2)),
                                "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_scale = make_tensor(act_dtype,
                                    layout_auto((bn, k_tile * 2), (1, 0)),
                                    "register")
                tr_bias = make_tensor(act_dtype,
                                    layout_auto((bn, k_tile * 2), (1, 0)),
                                    "register")

                # Set up global memory tensor views with custom address calculation
                tg_a = tensor_view(
                    a[:, k_start_pos:],
                    ComposedTensorLayout(
                        TensorLayout(
                            (bm, in_features),
                            (in_features,
                            1)),  # reference layout for instruction selection
                        0,  # base address
                        functor=functools.partial(
                            functor_a,
                            expert_lut=sorted_topk_ids[pid_m * bm:]),
                    ),
                    "global",
                )
                tg_b = tensor_view(b[expert_idx, pid_n * bn:, k_start_ofs:],
                                gmem_layout, "global")

                # Set up global memory views for scale and bias
                tg_scale = tensor_view(
                    scale[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )
                tg_bias = tensor_view(
                    bias[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )

                # Set up tensor partitions for memory operations
                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                txgbi = partition_src(tg_bias, auto_copy())
                txsbi = partition_dst(ts_bias, auto_copy())

                # Create masks for boundary conditions
                msk_a = mask(
                    auto_copy(),
                    [
                        expert_total_tokens - (pid_m - expert_start_pid) * bm,
                        i32(bk)
                    ],
                )
                msk_b = mask(auto_copy(), [out_features - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(),
                                [out_features - pid_n * bn,
                                i32(bk)])

                # Calculate number of K blocks
                k_block_max = (in_features + bk - 1) // bk

                # Pipeline stage 1: Load initial data into shared memory
                prologue_pipes = k_block_max if k_pipe_max > k_block_max else k_pipe_max
                for s in range(prologue_pipes):
                    # Copy data from global to shared memory with masking
                    copy(
                        auto_copy((bm, bk)),
                        txga[:, :, s],
                        txsa[:, :, s],
                        msk_a,
                        mbarrier=mbar_tma[s],
                    )
                    copy(
                        auto_copy((bn, bk)),
                        txgb[:, :, s],
                        txsb[:, :, s],
                        msk_b,
                        mbarrier=mbar_tma[s],
                        evict="evict_first",
                    )
                    copy(
                        auto_copy((bn, bk)),
                        txgsc[:, :, s],
                        txssc[:, :, s],
                        msk_scale,
                        mbarrier=mbar_tma[s],
                    )
                    copy(
                        auto_copy((bn, bk)),
                        txgbi[:, :, s],
                        txsbi[:, :, s],
                        msk_scale,
                        mbarrier=mbar_tma[s],
                    )
                    mbarrier_arrive(mbar_tma[s], tma_copy_tx)

                # Initialize pipeline stage tracking
                smem_pipe_read = 0
                smem_pipe_write = prologue_pipes if prologue_pipes < k_pipe_max else 0
                smem_pipe_release = 0
                read_phase = False
                release_phase = False

                # Set up tensor partitions for computation
                txSa = partition_A(ts_a, tiled_mma)

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())

                txSbi = partition_src(ts_bias, auto_copy())
                txrbi = partition_dst(tr_bias, auto_copy())

                # Main computation loop with pipeline stages
                k_tile_max = bk // k_tile

                # Pipeline stage 2: Initial MMA computation
                wgmma_fence_operand(tr_c)
                prologue_mmas = (k_block_max
                                if k_pipe_mmas > k_block_max else k_pipe_mmas)
                for ko in range(prologue_mmas):
                    mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                    # Load data for current stage
                    copy(auto_copy(), txSb[:, :, 0, smem_pipe_read], txrb[:, :,
                                                                        0])
                    copy(auto_copy(), txSsc[:, :, 0, smem_pipe_read],
                        txrsc[:, :, 0])
                    copy(auto_copy(), txSbi[:, :, 0, smem_pipe_read],
                        txrbi[:, :, 0])

                    # Process each tile in the current stage
                    for ki in range(k_tile_max):
                        # Dequantize weights and perform MMA
                        txrb_f16 = txrsc[:, :, ki % 2] * (
                            cast(txrb[:, :, ki % 2], act_dtype) -
                            txrbi[:, :, ki % 2])
                        wgmma_fence()
                        mma(
                            tiled_mma,
                            tr_c,
                            txSa[:, :, ki, smem_pipe_read],
                            txrb_f16,
                            tr_c,
                        )

                        # Prepare next tile if not the last one
                        if ki < k_tile_max - 1:
                            copy(
                                auto_copy(),
                                txSb[:, :, ki + 1, smem_pipe_read],
                                txrb[:, :, (ki + 1) % 2],
                            )
                            copy(
                                auto_copy(),
                                txSsc[:, :, ki + 1, smem_pipe_read],
                                txrsc[:, :, (ki + 1) % 2],
                            )
                            copy(
                                auto_copy(),
                                txSbi[:, :, ki + 1, smem_pipe_read],
                                txrbi[:, :, (ki + 1) % 2],
                            )
                    wgmma_commit_group()
                    smem_pipe_read += 1
                wgmma_fence_operand(tr_c)

                # Pipeline stage 3: Remaining MMA computation with synchronization
                k_blocks = (k_block_max - prologue_mmas
                            if k_block_max > prologue_mmas else 0)
                for ko in range(k_blocks):
                    mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)

                    wgmma_fence_operand(tr_c)
                    # Load data for current stage
                    copy(auto_copy(), txSb[:, :, 0, smem_pipe_read], txrb[:, :,
                                                                        0])
                    copy(auto_copy(), txSsc[:, :, 0, smem_pipe_read],
                        txrsc[:, :, 0])
                    copy(auto_copy(), txSbi[:, :, 0, smem_pipe_read],
                        txrbi[:, :, 0])

                    # Process each tile
                    for ki in range(k_tile_max):
                        # Dequantize weights and perform MMA
                        txrb_f16 = txrsc[:, :, ki % 2] * (
                            cast(txrb[:, :, ki % 2], act_dtype) -
                            txrbi[:, :, ki % 2])
                        wgmma_fence()
                        mma(
                            tiled_mma,
                            tr_c,
                            txSa[:, :, ki, smem_pipe_read],
                            txrb_f16,
                            tr_c,
                        )

                        # Prepare next tile if not the last one
                        if ki < k_tile_max - 1:
                            copy(
                                auto_copy(),
                                txSb[:, :, ki + 1, smem_pipe_read],
                                txrb[:, :, (ki + 1) % 2],
                            )
                            copy(
                                auto_copy(),
                                txSsc[:, :, ki + 1, smem_pipe_read],
                                txrsc[:, :, (ki + 1) % 2],
                            )
                            copy(
                                auto_copy(),
                                txSbi[:, :, ki + 1, smem_pipe_read],
                                txrbi[:, :, (ki + 1) % 2],
                            )
                    wgmma_commit_group()
                    wgmma_fence_operand(tr_c)

                    # Pipeline synchronization using mbarriers
                    wgmma_wait_group(k_pipe_mmas)
                    mbarrier_arrive(mbar_mma[smem_pipe_release])

                    # Load next block of data if available
                    if ko + k_pipe_max < k_block_max:
                        mbarrier_wait(mbar_mma[smem_pipe_release],
                                    release_phase)
                        copy(
                            auto_copy((bm, bk)),
                            txga[:, :, ko + k_pipe_max],
                            txsa[:, :, smem_pipe_write],
                            msk_a,
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgb[:, :, ko + k_pipe_max],
                            txsb[:, :, smem_pipe_write],
                            msk_b,
                            mbarrier=mbar_tma[smem_pipe_write],
                            evict="evict_first",
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgsc[:, :, ko + k_pipe_max],
                            txssc[:, :, smem_pipe_write],
                            msk_scale,
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgbi[:, :, ko + k_pipe_max],
                            txsbi[:, :, smem_pipe_write],
                            msk_scale,
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        mbarrier_arrive(mbar_tma[smem_pipe_write], tma_copy_tx)
                        smem_pipe_write += 1
                        if smem_pipe_write == k_pipe_max:
                            smem_pipe_write = 0

                    # Update pipeline stage tracking
                    smem_pipe_read += 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = 0
                        read_phase = not read_phase
                    smem_pipe_release += 1
                    if smem_pipe_release == k_pipe_max:
                        smem_pipe_release = 0
                        release_phase = not release_phase

                # Wait for all MMA operations to complete
                wgmma_wait_group(0)
                wgmma_fence_operand(tr_c)

                # Apply routing weights if enabled
                if mul_routed_weight:
                    # Set up global memory view for routing weights
                    tg_w = tensor_view(
                        routed_weight,
                        ComposedTensorLayout(
                            TensorLayout((bm, bn), (out_features, 0)),
                            0,
                            functor=functools.partial(
                                functor_rw,
                                expert_lut=sorted_topk_ids[pid_m * bm:]),
                        ),
                        "global",
                    )
                    # Load and apply routing weights
                    tr_w = make_tensor(f32, auto_layout, "register")
                    txgw = partition_src(tg_w, auto_copy())
                    txrw = partition_dst(tr_w, auto_copy())
                    mask_w = mask(
                        auto_copy(),
                        [
                            expert_total_tokens -
                            (pid_m - expert_start_pid) * bm,
                            i32(bn),
                        ],
                    )
                    copy(auto_copy((bm, bn)), txgw, txrw, mask_w)
                    tr_c = tr_c * tr_w

                # Write results back to global memory
                msk_c = mask(
                    auto_copy(),
                    [
                        expert_total_tokens - (pid_m - expert_start_pid) * bm,
                        out_features - pid_n * bn,
                    ],
                )
                tr_C = rearrange(cast(tr_c, act_dtype), auto_layout,
                                "register")
                tg_c = tensor_view(
                    c[:, pid_n * bn:],
                    ComposedTensorLayout(
                        TensorLayout(
                            (bm, bn),
                            (out_features,
                            1)),  # reference layout for instruction selection
                        0,  # base address
                        functor=functools.partial(
                            functor_c,
                            expert_lut=sorted_topk_ids[pid_m * bm:]),
                    ),
                    "global",
                )

                # Copy results to global memory with masking
                txrx_c = partition_src(tr_C, auto_copy())
                txgx_c = partition_dst(tg_c, auto_copy())
                copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

        return script_module.ir_module()

    def _moe_wna16_warp_specialized(self, config: MoEConfig):
        (
            experts_per_token,
            in_features,
            out_features,
            num_experts,
            group_size,
            param_dtype,
            act_dtype,
        ) = self.unpack_parameters()

        from hidet.ir.cute.ops import (make_mbarriers, mbarrier_arrive,
                                       mbarrier_wait, partition_A,
                                       wgmma_fence_operand)

        # Get configuration parameters for tiled matrix multiplication
        tiled_mma = config.tiled_mma
        bk = config.block_k  # Block size in K dimension
        k_pipe_max = config.stages  # Number of pipeline stages
        bm, bn, _ = config.thread_block_shape  # Block sizes in M and N dimensions
        threads = config.threads  # Number of threads per block
        k_tile = config.k_tile  # Tile size for K dimension

        num_consumer_threads = threads
        num_producer_threads = 128

        if num_consumer_threads == 256:
            producer_warpgroups = [2]
            consumer_warpgroups = [0, 1]
        elif num_consumer_threads == 128:
            producer_warpgroups = [1]
            consumer_warpgroups = [0]
 
        # Define memory layouts optimized for H100
        # These layouts ensure coalesced memory access and avoid bank conflicts
        gmem_layout, smem_layout = deduce_gmem_layout(in_features, bk, bn,
                                                    k_pipe_max)

        # Define layouts for scale and bias tensors
        scale_gmem_layout = TensorLayout(
            (bn, (group_size, in_features // group_size)),
            (1, (0, out_features)))

        # Define shared memory layouts for scale and bias with pipeline stages
        if bk > group_size:
            scale_smem_layout = TensorLayout(
                (bn, (group_size, bk // group_size), k_pipe_max),
                (1, (0, bn), bn * bk // group_size),
            )
        else:
            scale_smem_layout = TensorLayout((bn, bk, k_pipe_max), (1, 0, bn))

        # Set up address calculation functions for MoE routing
        divisor = experts_per_token if self.divide_by_experts_per_token else 1
        functor_a = functools.partial(a_addr_function,
                                    experts_per_token=divisor,
                                    features=in_features)
        functor_c = functools.partial(c_addr_function, features=out_features)
        functor_rw = functools.partial(rw_addr_function, features=out_features)

        # Calculate dimensions for padded tensors and blocks
        num_tokens = symbol_var("num_tokens")
        max_num_tokens_padded = num_tokens * experts_per_token + num_experts * (
            bm - 1)
        max_num_m_blocks = cdiv(max_num_tokens_padded, bm)
        mul_routed_weight = self.mul_routed_weight
        tune.check(in_features %
                bk == 0)  # Ensure block size divides input features
        tma_copy_tx = (
            (bm * bk) * act_dtype.nbytes + (bn * bk) * param_dtype.nbits // 8 +
            bn * ((bk + group_size - 1) // group_size) * act_dtype.nbytes * 2)
        unroll = f"u{k_pipe_max}"

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                    a: act_dtype[num_tokens, in_features],  # Input tensor
                    b: param_dtype[num_experts, out_features,
                                in_features],  # Quantized weights
                    c: act_dtype[num_tokens * experts_per_token,
                                out_features],  # Output tensor
                    scale: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Scale factors
                    bias: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Zero points
                    sorted_topk_ids: i32[
                        max_num_tokens_padded],  # Sorted expert indices
                    expert_ids: i32[
                        max_num_m_blocks],  # Expert ID lookup table
                    total_tokens_per_expert: i32[
                        num_experts],  # Token count per expert
                    expert_start_index: i32[
                        num_experts],  # Starting index per expert
                    num_tokens_post_pad: ~i32,  # Number of tokens after padding
                    routed_weight: ~f32,  # Optional routing weights
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = num_producer_threads + num_consumer_threads  # 12 warps total
                attrs.cuda.grid_dim = cdiv(max_num_tokens_padded, bm) * cdiv(
                    out_features, bn)
                attrs.cuda.min_blocks = 1
                attrs.cuda.dynamic_smem_bytes = 0

                # Initialize starting positions for K dimension
                k_start_pos = 0
                k_start_ofs = gmem_layout((0, k_start_pos))

                # Calculate thread block and grid dimensions
                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(max_num_tokens_padded, bm)
                num_pid_n = cdiv(out_features, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                num_tokens_post_pad_value = num_tokens_post_pad[0]
                if pid_m * bm >= num_tokens_post_pad_value:
                    return
                pid_n = (pid % num_pid_in_group) // group_size_m

                expert_idx = i64(expert_ids[pid_m])
                expert_start_pid = expert_start_index[expert_idx]
                expert_total_tokens = total_tokens_per_expert[expert_idx]

                # Initialize mbarriers for pipeline synchronization
                mbar_mma = make_mbarriers(k_pipe_max)
                mbar_tma = make_mbarriers(k_pipe_max)

                # Allocate shared memory tensors for pipeline stages
                ts_a = make_tensor(act_dtype, layout_auto(
                    (bm, bk, k_pipe_max)), "shared")
                ts_b = make_tensor(param_dtype, smem_layout, "shared")
                ts_scale = make_tensor(act_dtype, scale_smem_layout, "shared")
                ts_bias = make_tensor(act_dtype, scale_smem_layout, "shared")

                # Set up global memory tensor views with custom address calculation
                tg_a = tensor_view(
                    a[:, k_start_pos:],
                    ComposedTensorLayout(
                        TensorLayout(
                            (bm, in_features),
                            (in_features,
                            1)),  # reference layout for instruction selection
                        0,  # base address
                        functor=functools.partial(
                            functor_a,
                            expert_lut=sorted_topk_ids[pid_m * bm:]),
                    ),
                    "global",
                )
                tg_b = tensor_view(b[expert_idx, pid_n * bn:, k_start_ofs:],
                                gmem_layout, "global")

                # Set up global memory views for scale and bias
                tg_scale = tensor_view(
                    scale[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )
                tg_bias = tensor_view(
                    bias[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )
                
                # Calculate number of K blocks
                k_block_max = (in_features + bk - 1) // bk

                with warp_groups_producer(producer_warpgroups, num_regs=40):
                    # Pipeline control variables
                    smem_pipe_write = 0
                    write_phase = True

                    # Set up tensor partitions for memory operations
                    txga = partition_src(tg_a, auto_copy())
                    txsa = partition_dst(ts_a, auto_copy())

                    txgb = partition_src(tg_b, auto_copy())
                    txsb = partition_dst(ts_b, auto_copy())

                    txgsc = partition_src(tg_scale, auto_copy())
                    txssc = partition_dst(ts_scale, auto_copy())

                    txgbi = partition_src(tg_bias, auto_copy())
                    txsbi = partition_dst(ts_bias, auto_copy())

                    # Create masks for boundary conditions
                    msk_a = mask(
                        auto_copy(),
                        [
                            expert_total_tokens - (pid_m - expert_start_pid) * bm,
                            i32(bk)
                        ],
                    )
                    msk_b = mask(auto_copy(), [out_features - pid_n * bn, i32(bk)])
                    msk_scale = mask(auto_copy(),
                                    [out_features - pid_n * bn,
                                    i32(bk)])

                    # Pipeline stage 1: Load initial data into shared memory
                    for ko in range(k_block_max):
                        if ko >= k_pipe_max:
                            mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)
                        # Copy data from global to shared memory with masking
                        copy(
                            auto_copy((bm, bk)),
                            txga[:, :, ko],
                            txsa[:, :, smem_pipe_write],
                            msk_a,
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgb[:, :, ko],
                            txsb[:, :, smem_pipe_write],
                            msk_b,
                            mbarrier=mbar_tma[smem_pipe_write],
                            evict="evict_first",
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgsc[:, :, ko],
                            txssc[:, :, smem_pipe_write],
                            msk_scale,
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgbi[:, :, ko],
                            txsbi[:, :, smem_pipe_write],
                            msk_scale,
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        mbarrier_arrive(mbar_tma[smem_pipe_write], tma_copy_tx)
                        smem_pipe_write += 1
                        if smem_pipe_write == k_pipe_max:
                            smem_pipe_write = 0
                            write_phase = not write_phase
                    for ko in range(k_pipe_max):
                        mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)
                        smem_pipe_write += 1
                        if smem_pipe_write == k_pipe_max:
                            smem_pipe_write = 0
                            write_phase = not write_phase

                with warp_groups_consumer(consumer_warpgroups, num_regs=240):
                    # Initialize pipeline stage tracking
                    smem_pipe_read = 0
                    smem_pipe_release = 0
                    read_phase = False
                    release_phase = False

                    # Allocate register tensors for computation
                    tr_b = make_tensor(param_dtype, layout_auto((bn, k_tile * 2)),
                                    "register")
                    tr_c = make_tensor("float32", auto_layout, "register")
                    fill(tr_c, 0.0)

                    tr_scale = make_tensor(act_dtype,
                                        layout_auto((bn, k_tile * 2), (1, 0)),
                                        "register")
                    tr_bias = make_tensor(act_dtype,
                                        layout_auto((bn, k_tile * 2), (1, 0)),
                                        "register")

                    # Set up tensor partitions for computation
                    txSa = partition_A(ts_a, tiled_mma)

                    txSb = partition_src(ts_b, auto_copy())
                    txrb = partition_dst(tr_b, auto_copy())

                    txSsc = partition_src(ts_scale, auto_copy())
                    txrsc = partition_dst(tr_scale, auto_copy())

                    txSbi = partition_src(ts_bias, auto_copy())
                    txrbi = partition_dst(tr_bias, auto_copy())

                    # Main computation loop with pipeline stages
                    k_tile_max = bk // k_tile

                    # Pipeline stage 2: Initial MMA computation
                    wgmma_fence_operand(tr_c)
                    mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                    copy(auto_copy(), txSb[:, :, 0, smem_pipe_read], txrb[:, :, 0])
                    copy(auto_copy(), txSsc[:, :, 0, smem_pipe_read], txrsc[:, :, 0])
                    copy(auto_copy(), txSbi[:, :, 0, smem_pipe_read], txrbi[:, :, 0])
                    
                    for ki in grid(k_tile_max - 1, attrs="u+"):
                        copy(auto_copy(), txSb[:, :, ki + 1, smem_pipe_read], txrb[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSsc[:, :, ki + 1, smem_pipe_read], txrsc[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSbi[:, :, ki + 1, smem_pipe_read], txrbi[:, :, (ki + 1) % 2])
                        txrb_f16 = txrsc[:, :, ki % 2] * (cast(txrb[:, :, ki % 2], act_dtype) - txrbi[:, :, ki % 2])
                        wgmma_fence()
                        mma(
                            tiled_mma,
                            tr_c,
                            txSa[:, :, ki, smem_pipe_read],
                            txrb_f16,
                            tr_c,
                        )
                        wgmma_commit_group()
                    read_stage = smem_pipe_read
                    smem_pipe_read += 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = 0
                        read_phase = not read_phase
                    wgmma_wait_group(2)
                    last_tile = (k_tile_max - 1) % 2
                    txrb_f16 = txrsc[:, :, last_tile] * (cast(txrb[:, :, last_tile], act_dtype) - txrbi[:, :, last_tile])
                    wgmma_fence()
                    mma(
                        tiled_mma,
                        tr_c,
                        txSa[:, :, k_tile_max - 1, read_stage],
                        txrb_f16,
                        tr_c,
                    )
                    wgmma_commit_group()
                    
                    mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                    copy(auto_copy(), txSb[:, :, 0, smem_pipe_read], txrb[:, :, 0])
                    copy(auto_copy(), txSsc[:, :, 0, smem_pipe_read], txrsc[:, :, 0])
                    copy(auto_copy(), txSbi[:, :, 0, smem_pipe_read], txrbi[:, :, 0])
                    wgmma_wait_group(2)
                    wgmma_fence_operand(tr_c)
                    
                    for ko in grid(k_block_max - 2, attrs=unroll):
                        read_stage = smem_pipe_read
                        smem_pipe_read += 1
                        if smem_pipe_read == k_pipe_max:
                            smem_pipe_read = 0
                            read_phase = not read_phase
                        
                        wgmma_fence_operand(tr_c)
                        for ki in grid(k_tile_max, attrs="u+"):
                            if ki == k_tile_max - 1:
                                mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                                copy(auto_copy(), txSb[:, :, 0, smem_pipe_read], txrb[:, :, 0])
                                copy(auto_copy(), txSsc[:, :, 0, smem_pipe_read], txrsc[:, :, 0])
                                copy(auto_copy(), txSbi[:, :, 0, smem_pipe_read], txrbi[:, :, 0])
                            else:
                                copy(auto_copy(), txSb[:, :, ki + 1, read_stage], txrb[:, :, (ki + 1) % 2])
                                copy(auto_copy(), txSsc[:, :, ki + 1, read_stage], txrsc[:, :, (ki + 1) % 2])
                                copy(auto_copy(), txSbi[:, :, ki + 1, read_stage], txrbi[:, :, (ki + 1) % 2])
                            txrb_f16 = txrsc[:, :, ki % 2] * (cast(txrb[:, :, ki % 2], act_dtype) - txrbi[:, :, ki % 2])
                            wgmma_fence()
                            mma(
                                tiled_mma,
                                tr_c,
                                txSa[:, :, ki, read_stage],
                                txrb_f16,
                                tr_c,
                            )
                            wgmma_commit_group()
                            wgmma_wait_group(2)
                            
                            if ki == 1:
                                mbarrier_arrive(mbar_mma[smem_pipe_release])
                                smem_pipe_release += 1
                                if smem_pipe_release == k_pipe_max:
                                    smem_pipe_release = 0
                                    release_phase = not release_phase
                        wgmma_fence_operand(tr_c)
                    wgmma_fence_operand(tr_c)
                    
                    wgmma_fence_operand(tr_c)
                    for ki in grid(k_tile_max - 1, attrs="u+"):
                        copy(auto_copy(), txSb[:, :, ki + 1, smem_pipe_read], txrb[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSsc[:, :, ki + 1, smem_pipe_read], txrsc[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSbi[:, :, ki + 1, smem_pipe_read], txrbi[:, :, (ki + 1) % 2])
                        txrb_f16 = txrsc[:, :, ki % 2] * (cast(txrb[:, :, ki % 2], act_dtype) - txrbi[:, :, ki % 2])
                        wgmma_fence()
                        mma(
                            tiled_mma,
                            tr_c,
                            txSa[:, :, ki, smem_pipe_read],
                            txrb_f16,
                            tr_c,
                        )
                        wgmma_commit_group()
                        wgmma_wait_group(2)
                        
                        if ki == 1:
                            mbarrier_arrive(mbar_mma[smem_pipe_release])
                            smem_pipe_release += 1
                            if smem_pipe_release == k_pipe_max:
                                smem_pipe_release = 0
                                release_phase = not release_phase
                    txrb_f16 = txrsc[:, :, (k_tile_max - 1) % 2] * (cast(txrb[:, :, (k_tile_max - 1) % 2], act_dtype) - txrbi[:, :, (k_tile_max - 1) % 2])
                    wgmma_fence()
                    mma(
                        tiled_mma,
                        tr_c,
                        txSa[:, :, k_tile_max - 1, smem_pipe_read],
                        txrb_f16,
                        tr_c,
                    )
                    wgmma_fence_operand(tr_c)
                    wgmma_wait_group(0)
                    mbarrier_arrive(mbar_mma[smem_pipe_release])
                    # Apply routing weights if enabled
                    if mul_routed_weight:
                        # Set up global memory view for routing weights
                        tg_w = tensor_view(
                            routed_weight,
                            ComposedTensorLayout(
                                TensorLayout((bm, bn), (out_features, 0)),
                                0,
                                functor=functools.partial(
                                    functor_rw,
                                    expert_lut=sorted_topk_ids[pid_m * bm:]),
                            ),
                            "global",
                        )
                        # Load and apply routing weights
                        tr_w = make_tensor(f32, auto_layout, "register")
                        txgw = partition_src(tg_w, auto_copy())
                        txrw = partition_dst(tr_w, auto_copy())
                        mask_w = mask(
                            auto_copy(),
                            [
                                expert_total_tokens -
                                (pid_m - expert_start_pid) * bm,
                                i32(bn),
                            ],
                        )
                        copy(auto_copy((bm, bn)), txgw, txrw, mask_w)
                        tr_c = tr_c * tr_w

                    # Write results back to global memory
                    msk_c = mask(
                        auto_copy(),
                        [
                            expert_total_tokens - (pid_m - expert_start_pid) * bm,
                            out_features - pid_n * bn,
                        ],
                    )
                    tr_C = rearrange(cast(tr_c, act_dtype), auto_layout,
                                    "register")
                    tg_c = tensor_view(
                        c[:, pid_n * bn:],
                        ComposedTensorLayout(
                            TensorLayout(
                                (bm, bn),
                                (out_features,
                                1)),  # reference layout for instruction selection
                            0,  # base address
                            functor=functools.partial(
                                functor_c,
                                expert_lut=sorted_topk_ids[pid_m * bm:]),
                        ),
                        "global",
                    )

                    # Copy results to global memory with masking
                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

        return script_module.ir_module()


    def _moe_wna16(self, config: MoEConfig):
        # Unpack configuration parameters for the MoE layer
        (
            experts_per_token,
            in_features,
            out_features,
            num_experts,
            group_size,
            param_dtype,
            act_dtype,
        ) = self.unpack_parameters()

        # Get configuration parameters for tiled matrix multiplication
        tiled_mma = config.tiled_mma
        bk = config.block_k  # Block size in K dimension
        stages = config.stages  # Number of pipeline stages
        bm, bn, _ = config.thread_block_shape  # Block sizes in M and N dimensions
        threads = config.threads  # Number of threads per block
        k_tile = config.k_tile  # Tile size for K dimension

        # Define memory layouts for global and shared memory
        # These layouts are optimized for coalesced memory access and bank conflict avoidance
        gmem_layout, smem_layout = deduce_gmem_layout(in_features, bk, bn,
                                                    stages)

        # Define layouts for scale and bias tensors in global memory
        scale_gmem_layout = TensorLayout(
            (bn, (group_size, in_features // group_size)),
            (1, (0, out_features)))

        # Define layouts for scale and bias tensors in shared memory
        # Handle different cases based on block size and group size
        if bk > group_size:
            scale_smem_layout = TensorLayout(
                (bn, (group_size, bk // group_size), stages),
                (1, (0, bn), bn * bk // group_size),
            )
        else:
            scale_smem_layout = TensorLayout((bn, bk, stages), (1, 0, bn))

        # Set up address calculation functions for different tensors
        # These functions handle the complex memory access patterns for MoE routing
        divisor = experts_per_token if self.divide_by_experts_per_token else 1
        functor_a = functools.partial(a_addr_function,
                                    experts_per_token=divisor,
                                    features=in_features)
        functor_c = functools.partial(c_addr_function, features=out_features)
        functor_rw = functools.partial(rw_addr_function, features=out_features)

        # Calculate dimensions for padded tensors and blocks
        num_tokens = symbol_var("num_tokens")
        max_num_tokens_padded = num_tokens * experts_per_token + num_experts * (
            bm - 1)
        max_num_m_blocks = cdiv(max_num_tokens_padded, bm)
        mul_routed_weight = self.mul_routed_weight
        tune.check(in_features %
                bk == 0)  # Ensure block size divides input features
        unroll = f"u{stages}"
        
        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                    a: act_dtype[num_tokens, in_features],  # Input tensor
                    b: param_dtype[num_experts, out_features,
                                in_features],  # Quantized weights
                    c: act_dtype[num_tokens * experts_per_token,
                                out_features],  # Output tensor
                    scale: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Scale factors
                    bias: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Zero points
                    sorted_topk_ids: i32[
                        max_num_tokens_padded],  # Sorted expert indices
                    expert_ids: i32[
                        max_num_m_blocks],  # Expert ID lookup table
                    total_tokens_per_expert: i32[
                        num_experts],  # Token count per expert
                    expert_start_index: i32[
                        num_experts],  # Starting index per expert
                    num_tokens_post_pad: ~i32,  # Number of tokens after padding
                    routed_weight: ~f32,  # Optional routing weights
            ):
                # For each expert in the sorted_topk_ids, this kernel computes
                # c[sorted_topk_ids[m], :] = a[sorted_topk_ids[m] // experts_per_token, :] @ ((b[expert_ids[m], :, :]
                #                                          - zeros[expert_ids[m], :, :]) * scale[expert_ids[m], :, :])
                # where m is the index in the sorted_topk_ids (0 <= m < len(sorted_topk_ids))
                # a: [num_tokens, in_features], the hidden states of the input tokens
                # b: [num_experts, out_features, in_features], the quantized weights of the experts
                # c: [num_tokens * experts_per_token, out_features], the output hidden states of the tokens
                # scale: [num_experts, in_features // group_size, out_features], the scale of the quantized weights
                # bias: [num_experts, in_features // group_size, out_features], the zero point of the quantized weights
                # sorted_topk_ids: [max_num_tokens_padded], a sorted list of expert indices
                # expert_ids: [max_num_m_blocks], a lookup table that maps block id in m dimension to the expert id
                # total_tokens_per_expert: [num_experts], the total number of tokens of each expert
                # expert_start_index: [num_experts], the starting m block id of each expert in the sorted_topk_ids
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(max_num_tokens_padded, bm) * cdiv(
                    out_features, bn)
                attrs.cuda.dynamic_smem_bytes = 0

                # Initialize starting positions for K dimension
                k_start_pos = 0
                k_start_ofs = gmem_layout((0, k_start_pos))

                # Calculate thread block and grid dimensions
                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(max_num_tokens_padded, bm)
                num_pid_n = cdiv(out_features, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                num_tokens_post_pad_value = num_tokens_post_pad[0]
                if pid_m * bm >= num_tokens_post_pad_value:
                    return
                pid_n = (pid % num_pid_in_group) // group_size_m

                # Get expert information for this block
                expert_idx = i64(expert_ids[pid_m])
                expert_start_pid = expert_start_index[expert_idx]
                expert_total_tokens = total_tokens_per_expert[expert_idx]

                # Allocate shared memory tensors for tiled computation
                ts_a = make_tensor(
                    act_dtype,
                    TensorLayout((bm, bk, stages), (bk, 1, bm * bk)),
                    "shared",
                )
                ts_b = make_tensor(param_dtype, smem_layout, "shared")
                ts_scale = make_tensor(act_dtype, scale_smem_layout, "shared")
                ts_bias = make_tensor(act_dtype, scale_smem_layout, "shared")

                # Allocate register tensors for computation
                tr_a = make_tensor(act_dtype, layout_auto((bm, k_tile * 2)),
                                "register")
                tr_b = make_tensor(param_dtype, layout_auto((bn, k_tile * 2)),
                                "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_scale = make_tensor(act_dtype,
                                    layout_auto((bn, k_tile * 2), (1, 0)),
                                    "register")
                tr_bias = make_tensor(act_dtype,
                                    layout_auto((bn, k_tile * 2), (1, 0)),
                                    "register")

                # Set up global memory tensor views with custom address calculation
                tg_a = tensor_view(
                    a[:, k_start_pos:],
                    ComposedTensorLayout(
                        TensorLayout(
                            (bm, in_features),
                            (in_features,
                            1)),  # reference layout for instruction selection
                        0,  # base address
                        functor=functools.partial(
                            functor_a,
                            expert_lut=sorted_topk_ids[pid_m * bm:]),
                    ),
                    "global",
                )
                tg_b = tensor_view(b[expert_idx, pid_n * bn:, k_start_ofs:],
                                gmem_layout, "global")

                # Set up global memory views for scale and bias
                tg_scale = tensor_view(
                    scale[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )
                tg_bias = tensor_view(
                    bias[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )

                # Set up tensor partitions for memory operations
                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                txgbi = partition_src(tg_bias, auto_copy())
                txsbi = partition_dst(ts_bias, auto_copy())

                # Create masks for boundary conditions
                msk_a = mask(
                    auto_copy(),
                    [
                        expert_total_tokens - (pid_m - expert_start_pid) * bm,
                        i32(bk)
                    ],
                )
                msk_b = mask(auto_copy(), [out_features - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(),
                                [out_features - pid_n * bn,
                                i32(bk)])

                # Calculate number of K blocks
                k_block_max = (in_features + bk - 1) // bk

                # Pipeline stage 1: Load initial data into shared memory
                for s in range(stages - 1):
                    if s < k_block_max:
                        # Copy data from global to shared memory with masking
                        copy(auto_copy((bm, bk)), txga[:, :, s], txsa[:, :, s],
                            msk_a)
                        copy(
                            auto_copy((bn, bk)),
                            txgb[:, :, s],
                            txsb[:, :, s],
                            msk_b,
                            evict="evict_first",
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgsc[:, :, s],
                            txssc[:, :, s],
                            msk_scale,
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgbi[:, :, s],
                            txsbi[:, :, s],
                            msk_scale,
                        )
                    cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()

                # Initialize pipeline stage tracking
                smem_pipe_read = 0
                smem_pipe_write = stages - 1

                # Set up tensor partitions for computation
                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())

                txSbi = partition_src(ts_bias, auto_copy())
                txrbi = partition_dst(tr_bias, auto_copy())

                # Get pointers to current pipeline stage
                txSsc_p = txSsc[:, :, :, smem_pipe_read]
                txSbi_p = txSbi[:, :, :, smem_pipe_read]

                # Load initial data into registers
                copy(auto_copy(), txSa[:, :, 0, smem_pipe_read], txra[:, :, 0])
                copy(auto_copy(), txSb[:, :, 0, smem_pipe_read], txrb[:, :, 0])

                copy(auto_copy(), txSsc_p[:, :, 0], txrsc[:, :, 0])
                copy(auto_copy(), txSbi_p[:, :, 0], txrbi[:, :, 0])

                # Main computation loop over K blocks
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    for ki in grid(k_tile_max, attrs="u+"):
                        # Handle pipeline synchronization
                        if ki == k_tile_max - 1:
                            cp_async_wait_group(allow_on_fly_groups=stages - 2)
                            syncthreads()

                        # Prepare next tile
                        k_tile_next = (ki + 1) % k_tile_max
                        copy(
                            auto_copy(),
                            txSa[:, :, k_tile_next, smem_pipe_read],
                            txra[:, :, (ki + 1) % 2],
                        )
                        copy(
                            auto_copy(),
                            txSb[:, :, k_tile_next, smem_pipe_read],
                            txrb[:, :, (ki + 1) % 2],
                        )
                        copy(
                            auto_copy(),
                            txSsc[:, :, k_tile_next, smem_pipe_read],
                            txrsc[:, :, (ki + 1) % 2],
                        )
                        copy(
                            auto_copy(),
                            txSbi[:, :, k_tile_next, smem_pipe_read],
                            txrbi[:, :, (ki + 1) % 2],
                        )

                        # Load next block of data if needed
                        if ki == 0:
                            if ko + stages - 1 < k_block_max:
                                copy(
                                    auto_copy((bm, bk)),
                                    txga[:, :, ko + stages - 1],
                                    txsa[:, :, smem_pipe_write],
                                    msk_a,
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgb[:, :, ko + stages - 1],
                                    txsb[:, :, smem_pipe_write],
                                    msk_b,
                                    evict="evict_first",
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgsc[:, :, ko + stages - 1],
                                    txssc[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgbi[:, :, ko + stages - 1],
                                    txsbi[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        # Update pipeline read position
                        if ki == k_tile_max - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = (0 if smem_pipe_read == stages
                                            else smem_pipe_read)

                        # Perform quantized matrix multiplication
                        # 1. Dequantize weights using scale and bias
                        # 2. Perform matrix multiplication
                        txrb_f16 = txrsc[:, :, ki % 2] * (
                            cast(txrb[:, :, ki % 2], act_dtype) -
                            txrbi[:, :, ki % 2])
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_f16,
                            tr_c)

                # Apply routing weights if enabled
                if mul_routed_weight:
                    # Set up global memory view for routing weights
                    tg_w = tensor_view(
                        routed_weight,
                        ComposedTensorLayout(
                            TensorLayout((bm, bn), (out_features, 0)),
                            0,
                            functor=functools.partial(
                                functor_rw,
                                expert_lut=sorted_topk_ids[pid_m * bm:]),
                        ),
                        "global",
                    )
                    # Load and apply routing weights
                    tr_w = make_tensor(f32, auto_layout, "register")
                    txgw = partition_src(tg_w, auto_copy())
                    txrw = partition_dst(tr_w, auto_copy())
                    mask_w = mask(
                        auto_copy(),
                        [
                            expert_total_tokens -
                            (pid_m - expert_start_pid) * bm,
                            i32(bn),
                        ],
                    )
                    copy(auto_copy((bm, bn)), txgw, txrw, mask_w)
                    tr_c = tr_c * tr_w

                # Write results back to global memory
                msk_c = mask(
                    auto_copy(),
                    [
                        expert_total_tokens - (pid_m - expert_start_pid) * bm,
                        out_features - pid_n * bn,
                    ],
                )
                tr_C = rearrange(cast(tr_c, act_dtype), auto_layout,
                                "register")
                tg_c = tensor_view(
                    c[:, pid_n * bn:],
                    ComposedTensorLayout(
                        TensorLayout(
                            (bm, bn),
                            (out_features,
                            1)),  # reference layout for instruction selection
                        0,  # base address
                        functor=functools.partial(
                            functor_c,
                            expert_lut=sorted_topk_ids[pid_m * bm:]),
                    ),
                    "global",
                )

                # Copy results to global memory with masking
                txrx_c = partition_src(tr_C, auto_copy())
                txgx_c = partition_dst(tg_c, auto_copy())
                copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

        return script_module.ir_module()

    def _moe_wna16_triton_dataflow(self, config: MoEConfig):
        # Unpack configuration parameters for the MoE layer
        (
            experts_per_token,
            in_features,
            out_features,
            num_experts,
            group_size,
            param_dtype,
            act_dtype,
        ) = self.unpack_parameters()

        # Get configuration parameters for tiled matrix multiplication
        tiled_mma = config.tiled_mma
        bk = config.block_k  # Block size in K dimension
        stages = config.stages  # Number of pipeline stages
        bm, bn, _ = config.thread_block_shape  # Block sizes in M and N dimensions
        threads = config.threads  # Number of threads per block
        k_tile = config.k_tile  # Tile size for K dimension

        # Define memory layouts for global and shared memory
        # These layouts are optimized for coalesced memory access and bank conflict avoidance
        gmem_layout, smem_layout = deduce_gmem_layout(in_features, bk, bn,
                                                    stages)

        # Define layouts for scale and bias tensors in global memory
        scale_gmem_layout = TensorLayout(
            (bn, (group_size, in_features // group_size)),
            (1, (0, out_features)))

        # Define layouts for scale and bias tensors in shared memory
        # Handle different cases based on block size and group size
        if bk > group_size:
            scale_smem_layout = TensorLayout(
                (bn, (group_size, bk // group_size), stages),
                (1, (0, bn), bn * bk // group_size),
            )
        else:
            scale_smem_layout = TensorLayout((bn, bk, stages), (1, 0, bn))

        # Set up address calculation functions for different tensors
        # These functions handle the complex memory access patterns for MoE routing
        divisor = experts_per_token if self.divide_by_experts_per_token else 1
        functor_a = functools.partial(a_addr_function,
                                    experts_per_token=divisor,
                                    features=in_features)
        functor_c = functools.partial(c_addr_function, features=out_features)
        functor_rw = functools.partial(rw_addr_function, features=out_features)

        # Calculate dimensions for padded tensors and blocks
        num_tokens = symbol_var("num_tokens")
        max_num_tokens_padded = num_tokens * experts_per_token + num_experts * (
            bm - 1)
        max_num_m_blocks = cdiv(max_num_tokens_padded, bm)
        mul_routed_weight = self.mul_routed_weight
        tune.check(in_features %
                bk == 0)  # Ensure block size divides input features

        gmem_layout = TensorLayout((bn, in_features), (in_features, 1))
        b_pipe_stage = 2
        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                    a: act_dtype[num_tokens, in_features],  # Input tensor
                    b: param_dtype[num_experts, out_features,
                                in_features],  # Quantized weights
                    c: act_dtype[num_tokens * experts_per_token,
                                out_features],  # Output tensor
                    scale: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Scale factors
                    bias: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Zero points
                    sorted_topk_ids: i32[
                        max_num_tokens_padded],  # Sorted expert indices
                    expert_ids: i32[
                        max_num_m_blocks],  # Expert ID lookup table
                    total_tokens_per_expert: i32[
                        num_experts],  # Token count per expert
                    expert_start_index: i32[
                        num_experts],  # Starting index per expert
                    num_tokens_post_pad: ~i32,  # Number of tokens after padding
                    routed_weight: ~f32,  # Optional routing weights
            ):
                # For each expert in the sorted_topk_ids, this kernel computes
                # c[sorted_topk_ids[m], :] = a[sorted_topk_ids[m] // experts_per_token, :] @ ((b[expert_ids[m], :, :]
                #                                          - zeros[expert_ids[m], :, :]) * scale[expert_ids[m], :, :])
                # where m is the index in the sorted_topk_ids (0 <= m < len(sorted_topk_ids))
                # a: [num_tokens, in_features], the hidden states of the input tokens
                # b: [num_experts, out_features, in_features], the quantized weights of the experts
                # c: [num_tokens * experts_per_token, out_features], the output hidden states of the tokens
                # scale: [num_experts, in_features // group_size, out_features], the scale of the quantized weights
                # bias: [num_experts, in_features // group_size, out_features], the zero point of the quantized weights
                # sorted_topk_ids: [max_num_tokens_padded], a sorted list of expert indices
                # expert_ids: [max_num_m_blocks], a lookup table that maps block id in m dimension to the expert id
                # total_tokens_per_expert: [num_experts], the total number of tokens of each expert
                # expert_start_index: [num_experts], the starting m block id of each expert in the sorted_topk_ids
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(max_num_tokens_padded, bm) * cdiv(
                    out_features, bn)
                attrs.cuda.dynamic_smem_bytes = 0

                # Initialize starting positions for K dimension
                k_start_pos = 0
                k_start_ofs = gmem_layout((0, k_start_pos))

                # Calculate thread block and grid dimensions
                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(max_num_tokens_padded, bm)
                num_pid_n = cdiv(out_features, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                num_tokens_post_pad_value = num_tokens_post_pad[0]
                if pid_m * bm >= num_tokens_post_pad_value:
                    return
                pid_n = (pid % num_pid_in_group) // group_size_m

                # Get expert information for this block
                expert_idx = i64(expert_ids[pid_m])
                expert_start_pid = expert_start_index[expert_idx]
                expert_total_tokens = total_tokens_per_expert[expert_idx]

                # Allocate shared memory tensors for tiled computation
                ts_a = make_tensor(
                    act_dtype,
                    TensorLayout((bm, bk, stages), (bk, 1, bm * bk)),
                    "shared",
                )
                ts_b = make_tensor(act_dtype, layout_auto((bn, bk, b_pipe_stage)), "shared")
                ts_scale = make_tensor(act_dtype, scale_smem_layout, "shared")
                ts_bias = make_tensor(act_dtype, scale_smem_layout, "shared")

                # Allocate register tensors for computation
                tr_a = make_tensor(act_dtype, layout_auto((bm, k_tile * 2)), "register")
                tr_b1 = make_tensor(act_dtype, layout_auto((bn, k_tile * 2)), "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_b = make_tensor(param_dtype, layout_auto((bn, bk)), "register")
                tr_scale = make_tensor(act_dtype, layout_auto((bn, (group_size, bk//group_size)), (1, (0, bn*group_size))), "register")
                tr_bias = make_tensor(act_dtype, layout_auto((bn, (group_size, bk//group_size)), (1, (0, bn*group_size))), "register")

                # Set up global memory tensor views with custom address calculation
                tg_a = tensor_view(
                    a[:, k_start_pos:],
                    ComposedTensorLayout(
                        TensorLayout(
                            (bm, in_features),
                            (in_features,
                            1)),  # reference layout for instruction selection
                        0,  # base address
                        functor=functools.partial(
                            functor_a,
                            expert_lut=sorted_topk_ids[pid_m * bm:]),
                    ),
                    "global",
                )
                tg_b = tensor_view(b[expert_idx, pid_n * bn:, k_start_ofs:],
                                gmem_layout, "global")

                # Set up global memory views for scale and bias
                tg_scale = tensor_view(
                    scale[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )
                tg_bias = tensor_view(
                    bias[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )

                # Set up tensor partitions for memory operations
                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                txgbi = partition_src(tg_bias, auto_copy())
                txsbi = partition_dst(ts_bias, auto_copy())

                # Create masks for boundary conditions
                msk_a = mask(
                    auto_copy(),
                    [
                        expert_total_tokens - (pid_m - expert_start_pid) * bm,
                        i32(bk)
                    ],
                )
                msk_b = mask(auto_copy(), [out_features - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(),
                                [out_features - pid_n * bn,
                                i32(bk)])

                # Calculate number of K blocks
                k_block_max = (in_features + bk - 1) // bk

                b_read = 0
                b_write = 0
                copy(auto_copy((bn, bk)), txgb[:, :, 0], txrb)
                # Pipeline stage 1: Load initial data into shared memory
                for s in range(stages - 1):
                    if s < k_block_max:
                        # Copy data from global to shared memory with masking
                        copy(auto_copy((bm, bk)), txga[:, :, s], txsa[:, :, s],
                            msk_a)
                        copy(
                            auto_copy((bn, bk)),
                            txgsc[:, :, s],
                            txssc[:, :, s],
                            msk_scale,
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgbi[:, :, s],
                            txsbi[:, :, s],
                            msk_scale,
                        )
                    cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()
                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())
                txSbi = partition_src(ts_bias, auto_copy())
                txrbi = partition_dst(tr_bias, auto_copy())
                copy(auto_copy((bn, bk)), txSsc[:, :, 0], txrsc)
                copy(auto_copy((bn, bk)), txSbi[:, :, 0], txrbi)
                tr_b_f16 = txrsc * (cast(txrb, act_dtype) - txrbi)
                tBrB = partition_src(tr_b_f16, auto_copy())
                copy(auto_copy((bn, bk)), tBrB, txsb[:, :, b_write])
                b_write = 1 - b_write
                syncthreads()

                # Initialize pipeline stage tracking
                smem_pipe_read = 0
                smem_pipe_write = stages - 1

                # Set up tensor partitions for computation
                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb1 = partition_dst(tr_b1, auto_copy())

                # Get pointers to current pipeline stage
                # Load initial data into registers
                copy(auto_copy(), txSa[:, :, 0, smem_pipe_read], txra[:, :, 0])
                copy(auto_copy(), txSb[:, :, 0, b_read], txrb1[:, :, 0]) 

                # Main computation loop over K blocks
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    for ki in grid(k_tile_max, attrs="u+"):
                        # Handle pipeline synchronization
                        if ki == k_tile_max - 1:
                            cp_async_wait_group(allow_on_fly_groups=stages - 2)
                            syncthreads()
                            copy(auto_copy((bn, bk)), txSsc[:, :, smem_pipe_read], txrsc)
                            copy(auto_copy((bn, bk)), txSbi[:, :, smem_pipe_read], txrbi)
                            tr_b_f16_ = txrsc * (cast(txrb, act_dtype) - txrbi)
                            tBrB_ = partition_src(tr_b_f16_, auto_copy())
                            copy(auto_copy((bn, bk)), tBrB_, txsb[:, :, b_write])
                            b_write = 1 - b_write
                            b_read = 1 - b_read
                            syncthreads()

                        # Prepare next tile
                        k_tile_next = (ki + 1) % k_tile_max
                        copy(
                            auto_copy(),
                            txSa[:, :, k_tile_next, smem_pipe_read],
                            txra[:, :, (ki + 1) % 2],
                        )
                        copy(
                            auto_copy(),
                            txSb[:, :, k_tile_next, b_read],
                            txrb1[:, :, (ki + 1) % 2],
                        )

                        # Load next block of data if needed
                        if ki == 0:
                            if ko + 1 < k_block_max:
                                copy(
                                    auto_copy((bn, bk)),
                                    txgb[:, :, ko + 1],
                                    txrb,
                                    msk_b,
                                    evict="evict_first",
                                )
                            if ko + stages - 1 < k_block_max:
                                copy(
                                    auto_copy((bm, bk)),
                                    txga[:, :, ko + stages - 1],
                                    txsa[:, :, smem_pipe_write],
                                    msk_a,
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgsc[:, :, ko + stages - 1],
                                    txssc[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgbi[:, :, ko + stages - 1],
                                    txsbi[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        # Update pipeline read position
                        if ki == k_tile_max - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = (0 if smem_pipe_read == stages else smem_pipe_read)

                        # Perform quantized matrix multiplication
                        # 1. Dequantize weights using scale and bias
                        # 2. Perform matrix multiplication
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb1[:, :, ki % 2], tr_c)

                # Apply routing weights if enabled
                if mul_routed_weight:
                    # Set up global memory view for routing weights
                    tg_w = tensor_view(
                        routed_weight,
                        ComposedTensorLayout(
                            TensorLayout((bm, bn), (out_features, 0)),
                            0,
                            functor=functools.partial(
                                functor_rw,
                                expert_lut=sorted_topk_ids[pid_m * bm:]),
                        ),
                        "global",
                    )
                    # Load and apply routing weights
                    tr_w = make_tensor(f32, auto_layout, "register")
                    txgw = partition_src(tg_w, auto_copy())
                    txrw = partition_dst(tr_w, auto_copy())
                    mask_w = mask(
                        auto_copy(),
                        [
                            expert_total_tokens -
                            (pid_m - expert_start_pid) * bm,
                            i32(bn),
                        ],
                    )
                    copy(auto_copy((bm, bn)), txgw, txrw, mask_w)
                    tr_c = tr_c * tr_w

                # Write results back to global memory
                msk_c = mask(
                    auto_copy(),
                    [
                        expert_total_tokens - (pid_m - expert_start_pid) * bm,
                        out_features - pid_n * bn,
                    ],
                )
                tr_C = rearrange(cast(tr_c, act_dtype), auto_layout,
                                "register")
                tg_c = tensor_view(
                    c[:, pid_n * bn:],
                    ComposedTensorLayout(
                        TensorLayout(
                            (bm, bn),
                            (out_features,
                            1)),  # reference layout for instruction selection
                        0,  # base address
                        functor=functools.partial(
                            functor_c,
                            expert_lut=sorted_topk_ids[pid_m * bm:]),
                    ),
                    "global",
                )

                # Copy results to global memory with masking
                txrx_c = partition_src(tr_C, auto_copy())
                txgx_c = partition_dst(tg_c, auto_copy())
                copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

        return script_module.ir_module()


    def _moe_wna16_split_k(self, config: MoEConfig):
        # Unpack configuration parameters for the MoE layer
        (
            experts_per_token,
            in_features,
            out_features,
            num_experts,
            group_size,
            param_dtype,
            act_dtype,
        ) = self.unpack_parameters()
        tune.check(in_features > 4 * out_features)
        tune.check(not self.mul_routed_weight)

        # Get configuration parameters for tiled matrix multiplication
        tiled_mma = config.tiled_mma
        bk = config.block_k  # Block size in K dimension
        stages = config.stages  # Number of pipeline stages
        bm, bn, _ = config.thread_block_shape  # Block sizes in M and N dimensions
        threads = config.threads  # Number of threads per block
        k_tile = config.k_tile  # Tile size for K dimension
        parallel_k_parts = config.parallel_k_parts  # Number of parallel K parts
        tune.check(in_features % (bk * parallel_k_parts) == 0)
        k_chunk = in_features // (bk * parallel_k_parts) * bk

        # Define memory layouts for global and shared memory
        # These layouts are optimized for coalesced memory access and bank conflict avoidance
        gmem_layout, smem_layout = deduce_gmem_layout(in_features, bk, bn,
                                                    stages)

        # Define layouts for scale and bias tensors in global memory
        scale_gmem_layout = TensorLayout(
            (bn, (group_size, in_features // group_size)),
            (1, (0, out_features)))

        # Define layouts for scale and bias tensors in shared memory
        # Handle different cases based on block size and group size
        if bk > group_size:
            scale_smem_layout = TensorLayout(
                (bn, (group_size, bk // group_size), stages),
                (1, (0, bn), bn * bk // group_size),
            )
        else:
            scale_smem_layout = TensorLayout((bn, bk, stages), (1, 0, bn))

        # Set up address calculation functions for different tensors
        # These functions handle the complex memory access patterns for MoE routing
        divisor = experts_per_token if self.divide_by_experts_per_token else 1
        functor_a = functools.partial(a_addr_function,
                                    experts_per_token=divisor,
                                    features=in_features)
        functor_c = functools.partial(c_addr_function, features=out_features)
        functor_rw = functools.partial(rw_addr_function, features=out_features)

        # Calculate dimensions for padded tensors and blocks
        num_tokens = symbol_var("num_tokens")
        max_num_tokens_padded = num_tokens * experts_per_token + num_experts * (
            bm - 1)
        max_num_m_blocks = cdiv(max_num_tokens_padded, bm)
        mul_routed_weight = self.mul_routed_weight
        tune.check(in_features %
                bk == 0)  # Ensure block size divides input features
        grid_n = cdiv(out_features, bn)
        unroll = f"u{stages}"

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                    a: act_dtype[num_tokens, in_features],  # Input tensor
                    b: param_dtype[num_experts, out_features,
                                in_features],  # Quantized weights
                    c: act_dtype[num_tokens * experts_per_token,
                                out_features],  # Output tensor
                    scale: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Scale factors
                    bias: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Zero points
                    sorted_topk_ids: i32[
                        max_num_tokens_padded],  # Sorted expert indices
                    expert_ids: i32[
                        max_num_m_blocks],  # Expert ID lookup table
                    total_tokens_per_expert: i32[
                        num_experts],  # Token count per expert
                    expert_start_index: i32[
                        num_experts],  # Starting index per expert
                    num_tokens_post_pad: ~i32,  # Number of tokens after padding
                    routed_weight: ~f32,  # Optional routing weights
                    c_partial: act_dtype[parallel_k_parts, num_tokens * experts_per_token,
                                out_features],  # Partial output tensor
                    locks: i32[max_num_m_blocks, grid_n]
            ):
                # For each expert in the sorted_topk_ids, this kernel computes
                # c[sorted_topk_ids[m], :] = a[sorted_topk_ids[m] // experts_per_token, :] @ ((b[expert_ids[m], :, :]
                #                                          - zeros[expert_ids[m], :, :]) * scale[expert_ids[m], :, :])
                # where m is the index in the sorted_topk_ids (0 <= m < len(sorted_topk_ids))
                # a: [num_tokens, in_features], the hidden states of the input tokens
                # b: [num_experts, out_features, in_features], the quantized weights of the experts
                # c: [num_tokens * experts_per_token, out_features], the output hidden states of the tokens
                # scale: [num_experts, in_features // group_size, out_features], the scale of the quantized weights
                # bias: [num_experts, in_features // group_size, out_features], the zero point of the quantized weights
                # sorted_topk_ids: [max_num_tokens_padded], a sorted list of expert indices
                # expert_ids: [max_num_m_blocks], a lookup table that maps block id in m dimension to the expert id
                # total_tokens_per_expert: [num_experts], the total number of tokens of each expert
                # expert_start_index: [num_experts], the starting m block id of each expert in the sorted_topk_ids
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = max_num_m_blocks * grid_n * parallel_k_parts
                attrs.cuda.dynamic_smem_bytes = 0

                # Initialize starting positions for K dimension
                k_chunk_id = blockIdx.x % parallel_k_parts
                k_start_pos = k_chunk_id * k_chunk
                k_start_ofs = gmem_layout((0, k_start_pos))

                # Calculate thread block and grid dimensions
                group_size_m = 8
                pid = blockIdx.x // parallel_k_parts
                num_pid_m = cdiv(max_num_tokens_padded, bm)
                num_pid_n = cdiv(out_features, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                num_tokens_post_pad_value = num_tokens_post_pad[0]
                if pid_m * bm >= num_tokens_post_pad_value:
                    return
                pid_n = (pid % num_pid_in_group) // group_size_m

                # Get expert information for this block
                expert_idx = i64(expert_ids[pid_m])
                expert_start_pid = expert_start_index[expert_idx]
                expert_total_tokens = total_tokens_per_expert[expert_idx]

                # Allocate shared memory tensors for tiled computation
                ts_a = make_tensor(
                    act_dtype,
                    TensorLayout((bm, bk, stages), (bk, 1, bm * bk)),
                    "shared",
                )
                ts_b = make_tensor(param_dtype, smem_layout, "shared")
                ts_scale = make_tensor(act_dtype, scale_smem_layout, "shared")
                ts_bias = make_tensor(act_dtype, scale_smem_layout, "shared")

                # Allocate register tensors for computation
                tr_a = make_tensor(act_dtype, layout_auto((bm, k_tile * 2)),
                                "register")
                tr_b = make_tensor(param_dtype, layout_auto((bn, k_tile * 2)),
                                "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_scale = make_tensor(act_dtype,
                                    layout_auto((bn, k_tile * 2), (1, 0)),
                                    "register")
                tr_bias = make_tensor(act_dtype,
                                    layout_auto((bn, k_tile * 2), (1, 0)),
                                    "register")

                # Set up global memory tensor views with custom address calculation
                tg_a = tensor_view(
                    a[:, k_start_pos:],
                    ComposedTensorLayout(
                        TensorLayout(
                            (bm, in_features),
                            (in_features,
                            1)),  # reference layout for instruction selection
                        0,  # base address
                        functor=functools.partial(
                            functor_a,
                            expert_lut=sorted_topk_ids[pid_m * bm:]),
                    ),
                    "global",
                )
                tg_b = tensor_view(b[expert_idx, pid_n * bn:, k_start_ofs:],
                                gmem_layout, "global")

                # Set up global memory views for scale and bias
                tg_scale = tensor_view(
                    scale[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )
                tg_bias = tensor_view(
                    bias[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )

                # Set up tensor partitions for memory operations
                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                txgbi = partition_src(tg_bias, auto_copy())
                txsbi = partition_dst(ts_bias, auto_copy())

                # Create masks for boundary conditions
                msk_a = mask(
                    auto_copy(),
                    [
                        expert_total_tokens - (pid_m - expert_start_pid) * bm,
                        i32(bk)
                    ],
                )
                msk_b = mask(auto_copy(), [out_features - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(),
                                [out_features - pid_n * bn,
                                i32(bk)])

                # Calculate number of K blocks
                k_block_max = (k_chunk + bk - 1) // bk

                # Pipeline stage 1: Load initial data into shared memory
                for s in range(stages - 1):
                    if s < k_block_max:
                        # Copy data from global to shared memory with masking
                        copy(auto_copy((bm, bk)), txga[:, :, s], txsa[:, :, s],
                            msk_a)
                        copy(
                            auto_copy((bn, bk)),
                            txgb[:, :, s],
                            txsb[:, :, s],
                            msk_b,
                            evict="evict_first",
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgsc[:, :, s],
                            txssc[:, :, s],
                            msk_scale,
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgbi[:, :, s],
                            txsbi[:, :, s],
                            msk_scale,
                        )
                    cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()

                # Initialize pipeline stage tracking
                smem_pipe_read = 0
                smem_pipe_write = stages - 1

                # Set up tensor partitions for computation
                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())

                txSbi = partition_src(ts_bias, auto_copy())
                txrbi = partition_dst(tr_bias, auto_copy())

                # Get pointers to current pipeline stage
                txSsc_p = txSsc[:, :, :, smem_pipe_read]
                txSbi_p = txSbi[:, :, :, smem_pipe_read]

                # Load initial data into registers
                copy(auto_copy(), txSa[:, :, 0, smem_pipe_read], txra[:, :, 0])
                copy(auto_copy(), txSb[:, :, 0, smem_pipe_read], txrb[:, :, 0])

                copy(auto_copy(), txSsc_p[:, :, 0], txrsc[:, :, 0])
                copy(auto_copy(), txSbi_p[:, :, 0], txrbi[:, :, 0])

                # Main computation loop over K blocks
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    for ki in grid(k_tile_max, attrs="u+"):
                        # Handle pipeline synchronization
                        if ki == k_tile_max - 1:
                            cp_async_wait_group(allow_on_fly_groups=stages - 2)
                            syncthreads()

                        # Prepare next tile
                        k_tile_next = (ki + 1) % k_tile_max
                        copy(
                            auto_copy(),
                            txSa[:, :, k_tile_next, smem_pipe_read],
                            txra[:, :, (ki + 1) % 2],
                        )
                        copy(
                            auto_copy(),
                            txSb[:, :, k_tile_next, smem_pipe_read],
                            txrb[:, :, (ki + 1) % 2],
                        )
                        copy(
                            auto_copy(),
                            txSsc[:, :, k_tile_next, smem_pipe_read],
                            txrsc[:, :, (ki + 1) % 2],
                        )
                        copy(
                            auto_copy(),
                            txSbi[:, :, k_tile_next, smem_pipe_read],
                            txrbi[:, :, (ki + 1) % 2],
                        )

                        # Load next block of data if needed
                        if ki == 0:
                            if ko + stages - 1 < k_block_max:
                                copy(
                                    auto_copy((bm, bk)),
                                    txga[:, :, ko + stages - 1],
                                    txsa[:, :, smem_pipe_write],
                                    msk_a,
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgb[:, :, ko + stages - 1],
                                    txsb[:, :, smem_pipe_write],
                                    msk_b,
                                    evict="evict_first",
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgsc[:, :, ko + stages - 1],
                                    txssc[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgbi[:, :, ko + stages - 1],
                                    txsbi[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        # Update pipeline read position
                        if ki == k_tile_max - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = (0 if smem_pipe_read == stages
                                            else smem_pipe_read)

                        # Perform quantized matrix multiplication
                        # 1. Dequantize weights using scale and bias
                        # 2. Perform matrix multiplication
                        txrb_f16 = txrsc[:, :, ki % 2] * (
                            cast(txrb[:, :, ki % 2], act_dtype) -
                            txrbi[:, :, ki % 2])
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_f16,
                            tr_c)

                tr_C = rearrange(cast(tr_c, act_dtype), auto_layout,
                                "register")
                lock_ptr = ~locks[pid_m, pid_n]
                if k_chunk_id < parallel_k_parts - 1:
                    # Write results back to global memory
                    msk_c = mask(
                        auto_copy(),
                        [
                            expert_total_tokens - (pid_m - expert_start_pid) * bm,
                            out_features - pid_n * bn,
                        ],
                    )
                    tg_c = tensor_view(
                        c_partial[k_chunk_id, :, pid_n * bn:],
                        ComposedTensorLayout(
                            TensorLayout(
                                (bm, bn),
                                (out_features,
                                1)),  # reference layout for instruction selection
                            0,  # base address
                            functor=functools.partial(
                                functor_c,
                                expert_lut=sorted_topk_ids[pid_m * bm:]),
                        ),
                        "global",
                    )

                    # Copy results to global memory with masking
                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)
                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lock_ptr, 1, sem="acq_rel")
                else:
                    acquire_seq_semaphore(lock_ptr, k_chunk_id)
                    release_seq_semaphore(lock_ptr, 0)
                    # Write results back to global memory
                    msk_c = mask(
                        auto_copy(),
                        [
                            expert_total_tokens - (pid_m - expert_start_pid) * bm,
                            out_features - pid_n * bn,
                        ],
                    )
                    tr_c_partial = make_tensor(act_dtype, auto_layout, "register")
                    for chunk_id in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_partial[chunk_id, :, pid_n * bn:],
                            ComposedTensorLayout(
                                TensorLayout(
                                    (bm, bn),
                                    (out_features,
                                    1)),  # reference layout for instruction selection
                                0,  # base address
                                functor=functools.partial(
                                    functor_c,
                                    expert_lut=sorted_topk_ids[pid_m * bm:]),
                            ),
                            "global",
                        )

                        # Copy results to global memory with masking
                        txrx_c = partition_dst(tr_c_partial, auto_copy())
                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txgx_c, txrx_c, msk_c)
                        tr_C = tr_C + tr_c_partial
                    tg_c = tensor_view(
                        c[:, pid_n * bn:],
                        ComposedTensorLayout(
                            TensorLayout(
                                (bm, bn),
                                (out_features,
                                1)),  # reference layout for instruction selection
                            0,  # base address
                            functor=functools.partial(
                                functor_c,
                                expert_lut=sorted_topk_ids[pid_m * bm:]),
                        ),
                        "global",
                    )

                    # Copy results to global memory with masking
                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

        return script_module.ir_module()

    def _moe_wna16_bad_smem_layout(self, config: MoEConfig):
        # Unpack configuration parameters for the MoE layer
        (
            experts_per_token,
            in_features,
            out_features,
            num_experts,
            group_size,
            param_dtype,
            act_dtype,
        ) = self.unpack_parameters()

        # Get configuration parameters for tiled matrix multiplication
        tiled_mma = config.tiled_mma
        bk = config.block_k  # Block size in K dimension
        stages = config.stages  # Number of pipeline stages
        bm, bn, _ = config.thread_block_shape  # Block sizes in M and N dimensions
        threads = config.threads  # Number of threads per block
        k_tile = config.k_tile  # Tile size for K dimension

        # Define memory layouts for global and shared memory
        # These layouts are optimized for coalesced memory access and bank conflict avoidance
        gmem_layout, smem_layout = deduce_gmem_layout(in_features, bk, bn,
                                                    stages)

        # Define layouts for scale and bias tensors in global memory
        scale_gmem_layout = TensorLayout(
            (bn, (group_size, in_features // group_size)),
            (1, (0, out_features)))

        # Define layouts for scale and bias tensors in shared memory
        # Handle different cases based on block size and group size
        if bk > group_size:
            scale_smem_layout = TensorLayout(
                (bn, (group_size, bk // group_size), stages),
                (1, (0, bn), bn * bk // group_size),
            )
        else:
            scale_smem_layout = TensorLayout((bn, bk, stages), (1, 0, bn))

        # Set up address calculation functions for different tensors
        # These functions handle the complex memory access patterns for MoE routing
        divisor = experts_per_token if self.divide_by_experts_per_token else 1
        functor_a = functools.partial(a_addr_function,
                                    experts_per_token=divisor,
                                    features=in_features)
        functor_c = functools.partial(c_addr_function, features=out_features)
        functor_rw = functools.partial(rw_addr_function, features=out_features)

        # Calculate dimensions for padded tensors and blocks
        num_tokens = symbol_var("num_tokens")
        max_num_tokens_padded = num_tokens * experts_per_token + num_experts * (
            bm - 1)
        max_num_m_blocks = cdiv(max_num_tokens_padded, bm)
        mul_routed_weight = self.mul_routed_weight
        tune.check(in_features %
                bk == 0)  # Ensure block size divides input features
        gmem_layout = TensorLayout((bn, in_features), (in_features, 1))
        smem_layout = TensorLayout((bn, bk, stages), (bk, 1, bn * bk))

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                    a: act_dtype[num_tokens, in_features],  # Input tensor
                    b: param_dtype[num_experts, out_features,
                                in_features],  # Quantized weights
                    c: act_dtype[num_tokens * experts_per_token,
                                out_features],  # Output tensor
                    scale: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Scale factors
                    bias: act_dtype[num_experts, in_features // group_size,
                                    out_features],  # Zero points
                    sorted_topk_ids: i32[
                        max_num_tokens_padded],  # Sorted expert indices
                    expert_ids: i32[
                        max_num_m_blocks],  # Expert ID lookup table
                    total_tokens_per_expert: i32[
                        num_experts],  # Token count per expert
                    expert_start_index: i32[
                        num_experts],  # Starting index per expert
                    num_tokens_post_pad: ~i32,  # Number of tokens after padding
                    routed_weight: ~f32,  # Optional routing weights
            ):
                # For each expert in the sorted_topk_ids, this kernel computes
                # c[sorted_topk_ids[m], :] = a[sorted_topk_ids[m] // experts_per_token, :] @ ((b[expert_ids[m], :, :]
                #                                          - zeros[expert_ids[m], :, :]) * scale[expert_ids[m], :, :])
                # where m is the index in the sorted_topk_ids (0 <= m < len(sorted_topk_ids))
                # a: [num_tokens, in_features], the hidden states of the input tokens
                # b: [num_experts, out_features, in_features], the quantized weights of the experts
                # c: [num_tokens * experts_per_token, out_features], the output hidden states of the tokens
                # scale: [num_experts, in_features // group_size, out_features], the scale of the quantized weights
                # bias: [num_experts, in_features // group_size, out_features], the zero point of the quantized weights
                # sorted_topk_ids: [max_num_tokens_padded], a sorted list of expert indices
                # expert_ids: [max_num_m_blocks], a lookup table that maps block id in m dimension to the expert id
                # total_tokens_per_expert: [num_experts], the total number of tokens of each expert
                # expert_start_index: [num_experts], the starting m block id of each expert in the sorted_topk_ids
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(max_num_tokens_padded, bm) * cdiv(
                    out_features, bn)
                attrs.cuda.dynamic_smem_bytes = 0

                # Initialize starting positions for K dimension
                k_start_pos = 0
                k_start_ofs = gmem_layout((0, k_start_pos))

                # Calculate thread block and grid dimensions
                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(max_num_tokens_padded, bm)
                num_pid_n = cdiv(out_features, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                num_tokens_post_pad_value = num_tokens_post_pad[0]
                if pid_m * bm >= num_tokens_post_pad_value:
                    return
                pid_n = (pid % num_pid_in_group) // group_size_m

                # Get expert information for this block
                expert_idx = i64(expert_ids[pid_m])
                expert_start_pid = expert_start_index[expert_idx]
                expert_total_tokens = total_tokens_per_expert[expert_idx]

                # Allocate shared memory tensors for tiled computation
                ts_a = make_tensor(
                    act_dtype,
                    TensorLayout((bm, bk, stages), (bk, 1, bm * bk)),
                    "shared",
                )
                ts_b = make_tensor(param_dtype, smem_layout, "shared")
                ts_scale = make_tensor(act_dtype, scale_smem_layout, "shared")
                ts_bias = make_tensor(act_dtype, scale_smem_layout, "shared")

                # Allocate register tensors for computation
                tr_a = make_tensor(act_dtype, layout_auto((bm, k_tile * 2)),
                                "register")
                tr_b = make_tensor(param_dtype, layout_auto((bn, k_tile * 2)),
                                "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_scale = make_tensor(act_dtype,
                                    layout_auto((bn, k_tile * 2), (1, 0)),
                                    "register")
                tr_bias = make_tensor(act_dtype,
                                    layout_auto((bn, k_tile * 2), (1, 0)),
                                    "register")

                # Set up global memory tensor views with custom address calculation
                tg_a = tensor_view(
                    a[:, k_start_pos:],
                    ComposedTensorLayout(
                        TensorLayout(
                            (bm, in_features),
                            (in_features,
                            1)),  # reference layout for instruction selection
                        0,  # base address
                        functor=functools.partial(
                            functor_a,
                            expert_lut=sorted_topk_ids[pid_m * bm:]),
                    ),
                    "global",
                )
                tg_b = tensor_view(b[expert_idx, pid_n * bn:, k_start_ofs:],
                                gmem_layout, "global")

                # Set up global memory views for scale and bias
                tg_scale = tensor_view(
                    scale[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )
                tg_bias = tensor_view(
                    bias[expert_idx, k_start_pos // group_size:, pid_n * bn:],
                    scale_gmem_layout,
                    "global",
                )

                # Set up tensor partitions for memory operations
                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                txgbi = partition_src(tg_bias, auto_copy())
                txsbi = partition_dst(ts_bias, auto_copy())

                # Create masks for boundary conditions
                msk_a = mask(
                    auto_copy(),
                    [
                        expert_total_tokens - (pid_m - expert_start_pid) * bm,
                        i32(bk)
                    ],
                )
                msk_b = mask(auto_copy(), [out_features - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(),
                                [out_features - pid_n * bn,
                                i32(bk)])

                # Calculate number of K blocks
                k_block_max = (in_features + bk - 1) // bk

                # Pipeline stage 1: Load initial data into shared memory
                for s in range(stages - 1):
                    if s < k_block_max:
                        # Copy data from global to shared memory with masking
                        copy(auto_copy((bm, bk)), txga[:, :, s], txsa[:, :, s],
                            msk_a)
                        copy(
                            auto_copy((bn, bk)),
                            txgb[:, :, s],
                            txsb[:, :, s],
                            msk_b,
                            evict="evict_first",
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgsc[:, :, s],
                            txssc[:, :, s],
                            msk_scale,
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgbi[:, :, s],
                            txsbi[:, :, s],
                            msk_scale,
                        )
                    cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()

                # Initialize pipeline stage tracking
                smem_pipe_read = 0
                smem_pipe_write = stages - 1

                # Set up tensor partitions for computation
                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())

                txSbi = partition_src(ts_bias, auto_copy())
                txrbi = partition_dst(tr_bias, auto_copy())

                # Get pointers to current pipeline stage
                txSsc_p = txSsc[:, :, :, smem_pipe_read]
                txSbi_p = txSbi[:, :, :, smem_pipe_read]

                # Load initial data into registers
                copy(auto_copy(), txSa[:, :, 0, smem_pipe_read], txra[:, :, 0])
                copy(auto_copy(), txSb[:, :, 0, smem_pipe_read], txrb[:, :, 0])

                copy(auto_copy(), txSsc_p[:, :, 0], txrsc[:, :, 0])
                copy(auto_copy(), txSbi_p[:, :, 0], txrbi[:, :, 0])

                # Main computation loop over K blocks
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    for ki in grid(k_tile_max, attrs="u+"):
                        # Handle pipeline synchronization
                        if ki == k_tile_max - 1:
                            cp_async_wait_group(allow_on_fly_groups=stages - 2)
                            syncthreads()

                        # Prepare next tile
                        k_tile_next = (ki + 1) % k_tile_max
                        copy(
                            auto_copy(),
                            txSa[:, :, k_tile_next, smem_pipe_read],
                            txra[:, :, (ki + 1) % 2],
                        )
                        copy(
                            auto_copy(),
                            txSb[:, :, k_tile_next, smem_pipe_read],
                            txrb[:, :, (ki + 1) % 2],
                        )
                        copy(
                            auto_copy(),
                            txSsc[:, :, k_tile_next, smem_pipe_read],
                            txrsc[:, :, (ki + 1) % 2],
                        )
                        copy(
                            auto_copy(),
                            txSbi[:, :, k_tile_next, smem_pipe_read],
                            txrbi[:, :, (ki + 1) % 2],
                        )

                        # Load next block of data if needed
                        if ki == 0:
                            if ko + stages - 1 < k_block_max:
                                copy(
                                    auto_copy((bm, bk)),
                                    txga[:, :, ko + stages - 1],
                                    txsa[:, :, smem_pipe_write],
                                    msk_a,
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgb[:, :, ko + stages - 1],
                                    txsb[:, :, smem_pipe_write],
                                    msk_b,
                                    evict="evict_first",
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgsc[:, :, ko + stages - 1],
                                    txssc[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgbi[:, :, ko + stages - 1],
                                    txsbi[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        # Update pipeline read position
                        if ki == k_tile_max - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = (0 if smem_pipe_read == stages
                                            else smem_pipe_read)

                        # Perform quantized matrix multiplication
                        # 1. Dequantize weights using scale and bias
                        # 2. Perform matrix multiplication
                        txrb_f16 = txrsc[:, :, ki % 2] * (
                            cast(txrb[:, :, ki % 2], act_dtype) -
                            txrbi[:, :, ki % 2])
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_f16,
                            tr_c)

                # Apply routing weights if enabled
                if mul_routed_weight:
                    # Set up global memory view for routing weights
                    tg_w = tensor_view(
                        routed_weight,
                        ComposedTensorLayout(
                            TensorLayout((bm, bn), (out_features, 0)),
                            0,
                            functor=functools.partial(
                                functor_rw,
                                expert_lut=sorted_topk_ids[pid_m * bm:]),
                        ),
                        "global",
                    )
                    # Load and apply routing weights
                    tr_w = make_tensor(f32, auto_layout, "register")
                    txgw = partition_src(tg_w, auto_copy())
                    txrw = partition_dst(tr_w, auto_copy())
                    mask_w = mask(
                        auto_copy(),
                        [
                            expert_total_tokens -
                            (pid_m - expert_start_pid) * bm,
                            i32(bn),
                        ],
                    )
                    copy(auto_copy((bm, bn)), txgw, txrw, mask_w)
                    tr_c = tr_c * tr_w

                # Write results back to global memory
                msk_c = mask(
                    auto_copy(),
                    [
                        expert_total_tokens - (pid_m - expert_start_pid) * bm,
                        out_features - pid_n * bn,
                    ],
                )
                tr_C = rearrange(cast(tr_c, act_dtype), auto_layout,
                                "register")
                tg_c = tensor_view(
                    c[:, pid_n * bn:],
                    ComposedTensorLayout(
                        TensorLayout(
                            (bm, bn),
                            (out_features,
                            1)),  # reference layout for instruction selection
                        0,  # base address
                        functor=functools.partial(
                            functor_c,
                            expert_lut=sorted_topk_ids[pid_m * bm:]),
                    ),
                    "global",
                )

                # Copy results to global memory with masking
                txrx_c = partition_src(tr_C, auto_copy())
                txgx_c = partition_dst(tg_c, auto_copy())
                copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

        return script_module.ir_module()


def compile_fused_moe(
    moe_align,
    moe1: MoELinearWnA16,
    moe2: MoELinearWnA16,
    silu_and_mul: Optional[Callable[..., Any]] = None,
    moe_sum: Optional[Callable[..., Any]] = None,
):
    """Compile the fused MoE layer components.

    Args:
        moe_align: Function to align tokens based on expert routing
        moe1: First MoE linear layer
        moe2: Second MoE linear layer
        silu_and_mul: Optional SiLU activation and multiplication function
        moe_sum: Optional function to sum expert outputs

    Returns:
        Tuple of compiled moe1 and moe2 layers
    """
    # Unpack parameters from first layer
    (experts_per_token, k, n2, num_experts, group_size, param_dtype,
    act_dtype) = (moe1.unpack_parameters())

    # Unpack and validate parameters from second layer
    _, n, k_, _, _, _, _ = moe2.unpack_parameters()
    assert k == k_
    assert n2 == n * 2
    adtype = dtype_to_torch(act_dtype)

    with hidet.option.context():
        hidet.option.num_local_workers(1)
        # Extract IR modules
        modules1 = tune.extract_ir_modules(moe1.modules)
        modules2 = tune.extract_ir_modules(moe2.modules)

    # Generate all possible module pairs
    module_pairs = itertools.product(modules1, modules2)

    def validate(module_pair):
        m1, m2 = module_pair
        cfg1 = m1._tuning_kwargs["config"]
        bm1, _, _ = cfg1.thread_block_shape
        cfg2 = m2._tuning_kwargs["config"]
        bm2, _, _ = cfg2.thread_block_shape
        return bm1 == bm2

    module_pairs = list(filter(validate, module_pairs))
    print(f"module_pairs: {len(module_pairs)}")

    from hashlib import sha256
    from hidet.drivers import build_ir_module
    from hidet.runtime import load_compiled_module
    from tqdm import tqdm
    from hidet.utils.multiprocess import parallel_imap_2ndlevel
    
    def build_job(args):
        arg1, arg2 = args
        ir_module1, output_dir1 = arg1
        ir_module2, output_dir2 = arg2
        build_ir_module(ir_module1, output_dir1, target='cuda')
        build_ir_module(ir_module2, output_dir2, target='cuda')
        
    jobs = []
    for m1, m2 in module_pairs:
        hash_dir1 = sha256(str(m1).encode()).hexdigest()[:16]
        hash_dir2 = sha256(str(m2).encode()).hexdigest()[:16]
        output_dir1 = hidet.utils.cache_dir('ir_modules', hash_dir1)
        output_dir2 = hidet.utils.cache_dir('ir_modules', hash_dir2)
        jobs.append(((m1, output_dir1), (m2, output_dir2)))

    artifacts = []
    import time 
    start_time = time.perf_counter()
    if len(jobs) == 1:
        for job in jobs:
            build_job(job)
    else:
        for _ in tqdm(
            parallel_imap_2ndlevel(build_job, jobs, is_remote_allowed=True), desc="Compiling", total=len(jobs), ncols=80
        ):
            pass
    for job in jobs:
        ir_module1, output_dir1 = job[0]
        ir_module2, output_dir2 = job[1]
        func1 = load_compiled_module(output_dir1)
        func2 = load_compiled_module(output_dir2)
        artifacts.append(((ir_module1, func1), (ir_module2, func2)))
    end_time = time.perf_counter()
    print(f"Compiling time: {end_time - start_time} seconds")
        
    for num_tokens in MoELinearWnA16.M_BINS:
        (hidden_state, qweight1, scales1, zeros1, topk_weight,
        topk_ids) = (create_fake_tensors(
            num_tokens,
            experts_per_token,
            k,
            n2,
            num_experts,
            group_size,
            param_dtype,
            act_dtype,
        ))
        _, qweight2, scales2, zeros2 = create_fake_tensors(
            num_tokens,
            experts_per_token,
            n,
            k,
            num_experts,
            group_size,
            param_dtype,
            act_dtype,
            return_topk=False,
        )
        intermediate_cache1 = torch.empty((num_tokens * experts_per_token, n2),
                                        dtype=adtype,
                                        device=topk_ids.device)
        intermediate_cache2 = torch.empty((num_tokens * experts_per_token, n),
                                        dtype=adtype,
                                        device=topk_ids.device)
        intermediate_cache3 = torch.empty((num_tokens * experts_per_token, k),
                                        dtype=adtype,
                                        device=topk_ids.device)
        out_hidden_state = torch.empty((num_tokens, k),
                                    dtype=adtype,
                                    device=topk_ids.device)
        num_tokens_post_pad = torch.empty((1, ),
                                        dtype=torch.int32,
                                        device=topk_ids.device)
        total_tokens_per_expert = torch.empty((num_experts, ),
                                            dtype=torch.int32,
                                            device=topk_ids.device)
        expert_start_index = torch.empty((num_experts, ),
                                        dtype=torch.int32,
                                        device=topk_ids.device)

        min_time = None
        min_cfg1 = None
        min_cfg2 = None
        min_fused_moe1 = None
        min_fused_moe2 = None
        for (m1, fused_moe1), (m2, fused_moe2) in artifacts:
            cfg1 = m1._tuning_kwargs["config"]
            cfg2 = m2._tuning_kwargs["config"]
            bm, _, _ = cfg1.thread_block_shape
            bm_, _, _ = cfg2.thread_block_shape
            assert bm == bm_
            
            average_tokens_per_expert = experts_per_token * num_tokens / num_experts 
            min_num_tokens = (8 * num_experts) / experts_per_token
            if num_tokens <= min_num_tokens and bm >= 32:
                continue
            if num_tokens > min_num_tokens:
                if average_tokens_per_expert <= 256 and (bm < 0.5 * average_tokens_per_expert or bm > 2 * average_tokens_per_expert):
                    continue
                if average_tokens_per_expert > 256 and bm <= 64:
                    continue
            
            runtime_api.set_symbol_value("num_tokens", num_tokens)
            runtime_api.set_symbol_value("total_tokens",
                                        num_tokens * experts_per_token)
            runtime_api.set_symbol_value("block_size", bm)
            threads = FusedMoEWnA16.MOE_ALIGN_THREADS
            (max_num_tokens_padded, max_num_m_blocks) = (
                compute_max_num_tokens_padded_and_max_num_m_blocks(
                    num_tokens, experts_per_token, num_experts, bm))
            sorted_ids = torch.empty((max_num_tokens_padded, ),
                                    dtype=torch.int32,
                                    device=topk_ids.device)
            expert_ids = torch.empty((max_num_m_blocks, ),
                                    dtype=torch.int32,
                                    device=topk_ids.device)
            if cfg1.parallel_k_parts > 1:
                max_num_m_blocks = cdiv(max_num_tokens_padded, bm)
                grid_n = cdiv(n2, cfg1.block_n)
                locks1 = torch.zeros((max_num_m_blocks, grid_n),
                                    dtype=torch.int32,
                                    device=topk_ids.device)
                intermediate_cache3_partial = torch.empty((cfg1.parallel_k_parts, num_tokens * experts_per_token, n2),
                                    dtype=adtype,
                                    device=topk_ids.device)

            def fn():
                lock = torch.zeros((1, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
                tokens_cnt = torch.zeros(
                    (cdiv(num_tokens * experts_per_token,
                        threads), num_experts),
                    dtype=torch.int32,
                    device=topk_ids.device,
                )
                moe_align(
                    topk_ids,
                    tokens_cnt,
                    lock,
                    expert_ids,
                    sorted_ids,
                    total_tokens_per_expert,
                    expert_start_index,
                    num_tokens_post_pad,
                )
                if cfg1.parallel_k_parts > 1:
                    locks1.zero_()
                    fused_moe1(
                        hidden_state,
                        qweight1,
                        intermediate_cache1,
                        scales1,
                        zeros1,
                        sorted_ids,
                        expert_ids,
                        total_tokens_per_expert,
                        expert_start_index,
                        num_tokens_post_pad,
                        topk_weight,
                        intermediate_cache3_partial,
                        locks1,
                    )
                else:
                    fused_moe1(
                        hidden_state,
                        qweight1,
                        intermediate_cache1,
                        scales1,
                        zeros1,
                        sorted_ids,
                        expert_ids,
                        total_tokens_per_expert,
                        expert_start_index,
                        num_tokens_post_pad,
                        topk_weight,
                    )
                if silu_and_mul is not None:
                    silu_and_mul(intermediate_cache2, intermediate_cache1)
                fused_moe2(
                    intermediate_cache2,
                    qweight2,
                    intermediate_cache3,
                    scales2,
                    zeros2,
                    sorted_ids,
                    expert_ids,
                    total_tokens_per_expert,
                    expert_start_index,
                    num_tokens_post_pad,
                    topk_weight,
                )
                if moe_sum is not None:
                    moe_sum(out_hidden_state, intermediate_cache3)

            time = do_bench(fn, percentiles=None, warmup=5, rep=100, flush_l2_cache=True)
            # ruff: noqa: B023
            if min_time is None or time < min_time:
                min_time = time
                min_cfg1 = cfg1
                min_cfg2 = cfg2
                min_fused_moe1 = fused_moe1
                min_fused_moe2 = fused_moe2
        print(f"num_tokens: {num_tokens}, {min_cfg1}, {min_cfg2}")
        print(f"min_time: {min_time}")
        moe1.cache[num_tokens] = (min_cfg1, min_fused_moe1)
        moe2.cache[num_tokens] = (min_cfg2, min_fused_moe2)
    return moe1, moe2


class FusedMoEWnA16:
    """A fused Mixture of Experts layer combining two linear layers with activation.

    This class implements a fused MoE layer that combines:
    1. First linear layer with weight-only quantization
    2. SiLU activation and multiplication
    3. Second linear layer with weight-only quantization
    4. Expert routing and aggregation

    The fusion of these operations improves performance by reducing memory access
    and synchronization overhead.

    Attributes:
        MOE_ALIGN_THREADS: Number of threads used for MoE alignment
    """

    MOE_ALIGN_THREADS = 128

    def __init__(
        self,
        k: int,
        n: int,
        experts_per_token: int,
        num_experts: int,
        group_size: int,
        param_dtype: DataType,
        act_dtype: DataType,
        triton_dataflow: bool = False,
        triton_shared_memory: bool = False,
    ):
        """Initialize the fused MoE layer.

        Args:
            k: Input/output feature dimension
            n: Hidden dimension
            experts_per_token: Number of experts per token
            num_experts: Total number of experts
            group_size: Size of quantization groups
            param_dtype: Data type for parameters
            act_dtype: Data type for activations
        """
        self.k = k
        self.n = n
        self.experts_per_token = experts_per_token
        self.num_experts = num_experts
        self.group_size = group_size
        self.param_dtype = param_dtype
        self.act_dtype = data_type(act_dtype)

        # Initialize first MoE layer
        self.moe1 = MoELinearWnA16(experts_per_token, k, n * 2, num_experts,
                                group_size, param_dtype, act_dtype, triton_dataflow=triton_dataflow, triton_shared_memory=triton_shared_memory)

        # Initialize second MoE layer
        self.moe2 = MoELinearWnA16(
            experts_per_token,
            n,
            k,
            num_experts,
            group_size,
            param_dtype,
            act_dtype,
            divide_by_experts_per_token=False,
            mul_routed_weight=True,
            triton_dataflow=triton_dataflow,
            triton_shared_memory=triton_shared_memory,
        )

        # Initialize helper functions
        self.moe_align = moe_align_block_size_kernel(num_experts,
                                                    self.MOE_ALIGN_THREADS)
        self.silu_and_mul = silu_and_mul_kernel(n, act_dtype)
        self.moe_sum = moe_sum_kernel(experts_per_token, k, act_dtype)

        # Compile the fused layers
        compile_fused_moe(self.moe_align, self.moe1, self.moe2,
                        self.silu_and_mul, self.moe_sum)

    def __call__(
        self,
        hidden_state: torch.Tensor,
        qweight1: torch.Tensor,
        scales1: torch.Tensor,
        zeros1: torch.Tensor,
        qweight2: torch.Tensor,
        scales2: torch.Tensor,
        zeros2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        intermediate_cache1: Optional[torch.Tensor] = None,
        intermediate_cache2: Optional[torch.Tensor] = None,
        intermediate_cache3: Optional[torch.Tensor] = None,
        out_hidden_state: Optional[torch.Tensor] = None,
        return_intermediate_caches: Optional[bool] = False,
    ):
        """Execute the fused MoE layer.

        Args:
            hidden_state: Input hidden states [num_tokens, k]
            qweight1: Quantized weights for first layer [num_experts, n*2, k]
            scales1: Scale factors for first layer [num_experts, k//group_size, n*2]
            zeros1: Zero points for first layer [num_experts, k//group_size, n*2]
            qweight2: Quantized weights for second layer [num_experts, k, n]
            scales2: Scale factors for second layer [num_experts, n//group_size, k]
            zeros2: Zero points for second layer [num_experts, n//group_size, k]
            topk_ids: Top-k expert indices [num_tokens, experts_per_token]
            topk_weights: Top-k routing weights [num_tokens, experts_per_token]
            intermediate_cache1: Optional cache for first layer output
            intermediate_cache2: Optional cache for activation output
            intermediate_cache3: Optional cache for second layer output
            out_hidden_state: Optional output tensor
            return_intermediate_caches: Whether to return intermediate results

        Returns:
            Output hidden states [num_tokens, k] or tuple with intermediate results
            if return_intermediate_caches is True
        """
        # Get input dimensions
        num_tokens, _ = hidden_state.shape
        num_experts = self.num_experts
        experts_per_token = self.experts_per_token
        total_tokens = num_tokens * experts_per_token
        k = self.k
        n = self.n
        n2 = n * 2

        # Get configuration and block size
        config, _ = self.moe1.get_config(num_tokens)
        bm, _, _ = config.thread_block_shape

        # Calculate padded dimensions
        (max_num_tokens_padded, max_num_m_blocks) = (
            compute_max_num_tokens_padded_and_max_num_m_blocks(
                num_tokens, experts_per_token, num_experts, bm))

        # Initialize tensors for expert routing
        sorted_ids = torch.empty((max_num_tokens_padded, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
        expert_ids = torch.empty((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
        num_tokens_post_pad = torch.empty((1, ),
                                        dtype=torch.int32,
                                        device=topk_ids.device)
        total_tokens_per_expert = torch.empty((num_experts, ),
                                            dtype=torch.int32,
                                            device=topk_ids.device)
        expert_start_index = torch.empty((num_experts, ),
                                        dtype=torch.int32,
                                        device=topk_ids.device)

        # Initialize synchronization tensors
        lock = torch.zeros((1, ), dtype=torch.int32, device=topk_ids.device)
        tokens_cnt = torch.zeros(
            (cdiv(total_tokens, self.MOE_ALIGN_THREADS), num_experts),
            dtype=torch.int32,
            device=topk_ids.device,
        )

        # Set runtime values
        runtime_api.set_symbol_value("total_tokens", total_tokens)
        runtime_api.set_symbol_value("num_tokens", num_tokens)
        runtime_api.set_symbol_value("block_size", bm)

        # Align tokens based on expert routing
        self.moe_align(
            topk_ids,
            tokens_cnt,
            lock,
            expert_ids,
            sorted_ids,
            total_tokens_per_expert,
            expert_start_index,
            num_tokens_post_pad,
        )

        # Initialize intermediate caches if not provided
        adtype = dtype_to_torch(self.act_dtype)
        if intermediate_cache1 is None:
            intermediate_cache1 = torch.empty(
                (num_tokens * experts_per_token, n2),
                dtype=adtype,
                device=topk_ids.device,
            )
        if intermediate_cache2 is None:
            intermediate_cache2 = torch.empty(
                (num_tokens * experts_per_token, n),
                dtype=adtype,
                device=topk_ids.device,
            )
        if intermediate_cache3 is None:
            intermediate_cache3 = torch.empty(
                (num_tokens * experts_per_token, k),
                dtype=adtype,
                device=topk_ids.device,
            )
        if out_hidden_state is None:
            out_hidden_state = torch.empty((num_tokens, k),
                                        dtype=adtype,
                                        device=topk_ids.device)

        # Execute first MoE layer
        self.moe1(
            hidden_state,
            qweight1,
            intermediate_cache1,
            scales1,
            zeros1,
            sorted_ids,
            expert_ids,
            total_tokens_per_expert,
            num_tokens_post_pad,
            expert_start_index,
            topk_weights,
        )

        # Apply activation function
        self.silu_and_mul(intermediate_cache2, intermediate_cache1)

        # Execute second MoE layer
        self.moe2(
            intermediate_cache2,
            qweight2,
            intermediate_cache3,
            scales2,
            zeros2,
            sorted_ids,
            expert_ids,
            total_tokens_per_expert,
            num_tokens_post_pad,
            expert_start_index,
            topk_weights,
        )

        # Sum expert outputs
        self.moe_sum(out_hidden_state, intermediate_cache3)

        # Return results
        if return_intermediate_caches:
            return (
                out_hidden_state,
                intermediate_cache1,
                intermediate_cache2,
                intermediate_cache3,
            )
        else:
            return out_hidden_state


def fused_moe_wna16(
    k: int,
    n: int,
    experts_per_token: int,
    num_experts: int,
    group_size: int,
    param_dtype: str,
    act_dtype: str = "float16",
    triton_dataflow: bool = False,
    triton_shared_memory: bool = False,
):
    """Create a fused MoE layer with weight-only quantization.

    Args:
        k: Input/output feature dimension
        n: Hidden dimension
        experts_per_token: Number of experts per token
        num_experts: Total number of experts
        group_size: Size of quantization groups
        param_dtype: Data type for parameters
        act_dtype: Data type for activations

    Returns:
        A FusedMoEWnA16 instance
    """

    with hidet.option.context():
        hidet.option.search_space(2)
        hidet.option.use_torch_stream(True)
        moe = FusedMoEWnA16(k, n, experts_per_token, num_experts, group_size,
                            param_dtype, act_dtype, triton_dataflow, triton_shared_memory)
    return moe
