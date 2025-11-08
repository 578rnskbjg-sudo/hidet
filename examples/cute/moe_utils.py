# SPDX-License-Identifier: Apache-2.0
# type: ignore
# ruff: noqa
# type: ignore
"""
Helper functions for Mixture of Experts (MoE) layer implementation.
This module provides utilities for handling MoE operations, particularly focusing on
block size alignment and weight quantization/dequantization for efficient MoE computation.
"""

from typing import Union

import hidet
import torch
from hidet.ir.cute import TensorLayout, TiledTensorLayout
from hidet.ir.cute.algorithm import auto_copy
from hidet.ir.cute.layout import (Level, ThrValAtom, auto_layout, composition,
                                  layout_auto, logical_divide, make_layout)
from hidet.ir.cute.ops import (cast, copy, fill, make_tensor, mask,
                               partition_dst, partition_src, rearrange,
                               reduce_sum, silu, tensor_view)
from hidet.ir.dtypes import f16, f32, i32, i64, u4, u32
from hidet.ir.expr import symbol_var
from hidet.ir.primitives.cuda.atomic import atomic_add
from hidet.ir.primitives.cuda.mutex import acquire_seq_semaphore
from hidet.ir.type import DataType, data_type
from hidet.lang import attrs
from hidet.lang.cuda import (blockIdx, dynamic_shared_memory, syncthreads,
                             threadIdx)
from hidet.utils.py import cdiv, gcd


def moe_align_block_size_stage1(num_experts: int, threads: int = 256):
    """
    First stage of MoE block size alignment kernel.
    
    This kernel performs the initial counting and alignment of tokens per expert.
    It counts how many tokens are assigned to each expert and prepares the data
    for the second stage of alignment.
    
    Args:
        num_experts (int): Number of experts in the MoE layer
        threads (int, optional): Number of threads per block. Defaults to 256.
    
    Returns:
        Compiled CUDA kernel function
    """
    scalar_t = u32
    num_tokens = symbol_var("total_tokens")
    block_size = symbol_var("block_size")
    num_blocks = cdiv(num_tokens, threads)
    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)

    dynamic_smem_bytes = (num_experts * threads + num_experts + 1) * i32.nbytes

    accesses_per_threads = 4
    atom_shape = (1, accesses_per_threads)
    atom = TensorLayout(((1, ), (1, accesses_per_threads)), ((1, ), (1, 1)))
    tv_atom = ThrValAtom("thread", atom_shape, atom)

    thread_n = gcd(threads // accesses_per_threads, threads)
    thread_m = threads // thread_n
    repeat_n = 1
    repeat_m = num_experts // thread_m
    threads_in_thread_block = Level(
        "thread", "thread_block", (thread_m, thread_n),
        TensorLayout((thread_n, thread_m), (thread_m, 1)),
        (repeat_m, repeat_n))
    layout_tokens_cnt = TiledTensorLayout(tv_atom, [threads_in_thread_block])

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            topk_ids: scalar_t[num_tokens],
            token_cnts: i32[num_blocks, num_experts],
            lock: ~i32,
            expert_ids: i32[max_num_m_blocks],
            total_tokens_per_expert: i32[num_experts],
            expert_start_index: i32[num_experts],
            num_tokens_post_pad: ~i32,
        ):
            # Step 1: Initialize shared memory for token counts
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = num_blocks
            attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

            tid = threadIdx.x
            bid = blockIdx.x
            gid = threadIdx.x + blockIdx.x * threads

            # Allocate shared memory for token counts and cumulative sums
            smem_token_cnts = dynamic_shared_memory(byte_offset=0, dtype=i32)
            smem_cumsum = dynamic_shared_memory(byte_offset=num_experts *
                                                threads * 4,
                                                dtype=i32)

            # Step 2: Initialize token counts to zero
            for i in range(num_experts):
                smem_token_cnts[i * threads + tid] = 0
            syncthreads()

            # Step 3: Count tokens per expert
            if gid < num_tokens:
                expert_id = topk_ids[gid]
                smem_token_cnts[expert_id * threads + tid] += 1

            syncthreads()

            # Step 4: Reduce token counts across threads
            ts_token_cnt = tensor_view(smem_token_cnts,
                                       TensorLayout((num_experts, threads),
                                                    (threads, 1)),
                                       "shared",
                                       volatile=True)
            tr_token_cnt = make_tensor(i32, layout_tokens_cnt, "register")
            txstoken_cnt = partition_src(ts_token_cnt, auto_copy())
            txrtoken_cnt = partition_dst(tr_token_cnt, auto_copy())
            copy(auto_copy((num_experts, threads)), txstoken_cnt, txrtoken_cnt)

            syncthreads()

            # Step 5: Sum up token counts
            tr_token_cnt_sum = reduce_sum(tr_token_cnt, 1)

            # Step 6: Handle block synchronization and accumulation
            if bid > 0:
                # For non-first blocks, accumulate counts to global memory
                tg_token_cnt = tensor_view(
                    token_cnts[bid, :],
                    TensorLayout((num_experts, threads), (1, 0)), "global")
                txgtoken_cnt = partition_dst(tg_token_cnt, auto_copy())
                txrtoken_cnt_sum = partition_src(tr_token_cnt_sum, auto_copy())
                copy(auto_copy((num_experts, threads)), txrtoken_cnt_sum,
                     txgtoken_cnt)
                syncthreads()

                if tid == 0:
                    atomic_add(lock, 1, sem="acq_rel")
            else:
                # For first block, handle synchronization and compute final indices
                acquire_seq_semaphore(lock, num_blocks - 1)

                # Accumulate counts from other blocks
                for i in range(num_blocks - 1):
                    tr_token_cnt_partial = make_tensor(
                        i32, layout_auto((num_experts, threads), (1, 0)),
                        "register")
                    tg_token_cnt_partial = tensor_view(
                        token_cnts[i + 1, :],
                        TensorLayout((num_experts, threads), (1, 0)), "global")
                    txgtoken_cnt_partial = partition_src(
                        tg_token_cnt_partial, auto_copy())
                    txrtoken_cnt_partial = partition_dst(
                        tr_token_cnt_partial, auto_copy())
                    copy(auto_copy((num_experts, threads)),
                         txgtoken_cnt_partial, txrtoken_cnt_partial)
                    tr_token_cnt_sum = tr_token_cnt_partial + tr_token_cnt_sum

                # Compute cumulative sums and expert indices
                ts_cumsum = tensor_view(smem_cumsum + 1,
                                        TensorLayout((num_experts, threads),
                                                     (1, 0)),
                                        "shared",
                                        volatile=True)
                txscumsum = partition_src(ts_cumsum, auto_copy())
                txrcumsum = partition_dst(tr_token_cnt_sum, auto_copy())
                copy(auto_copy((num_experts, threads)), txrcumsum, txscumsum)

                syncthreads()

                # Step 7: Compute final indices and padding
                if tid == 0:
                    smem_cumsum[0] = 0
                    for i in range(num_experts):
                        total_tokens_per_expert[i] = smem_cumsum[i + 1]
                        expert_start_index[i] = smem_cumsum[i] // block_size
                        smem_cumsum[i + 1] = smem_cumsum[i] + (smem_cumsum[
                            i + 1] + block_size - 1) // block_size * block_size
                    num_tokens_post_pad[0] = smem_cumsum[num_experts]

                syncthreads()

                # Step 8: Generate expert IDs for each block
                rounds = cdiv(num_experts, threads)
                for i in range(rounds):
                    eid = i * threads + tid
                    if eid < num_experts:
                        beg = smem_cumsum[eid] // block_size
                        end = smem_cumsum[eid + 1] // block_size
                        for j in range(end - beg):
                            expert_ids[j + beg] = eid

    func = script_module.build()
    return func


def moe_align_block_size_stage2(num_experts: int, threads: int = 256):
    """
    Second stage of MoE block size alignment kernel.
    
    This kernel performs the final sorting and alignment of tokens based on expert assignments.
    It uses the results from stage 1 to place tokens in their correct positions in the sorted buffer.
    
    Args:
        num_experts (int): Number of experts in the MoE layer
        threads (int, optional): Number of threads per block. Defaults to 256.
    
    Returns:
        Compiled CUDA kernel function
    """
    scalar_t = u32
    num_tokens = symbol_var("total_tokens")
    block_size = symbol_var("block_size")
    num_blocks = cdiv(num_tokens, threads)
    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            topk_ids: scalar_t[num_tokens],
            token_cnts: i32[num_blocks, num_experts],
            expert_start_index: i32[num_experts],
            sorted_topk_ids: i32[max_num_tokens_padded],
        ):
            # Step 1: Set up kernel parameters and thread indices
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = num_blocks
            attrs.cuda.dynamic_smem_bytes = 0

            bid = blockIdx.x
            gid = threadIdx.x + blockIdx.x * threads

            # Step 2: Sort tokens by expert and compute their positions
            if gid < num_tokens:
                expert_id = topk_ids[gid]
                pos = expert_start_index[expert_id] * block_size
                ticket = atomic_add(~token_cnts[0, expert_id], 1)
                sorted_topk_ids[pos + ticket] = gid

    func = script_module.build()
    return func


class MoEAlignBlockSize:
    """
    A class that implements the two-stage MoE block size alignment process.
    
    This class combines the two stages of MoE block size alignment into a single interface.
    It manages the execution of both stages and handles the necessary data transfers between them.
    
    Args:
        num_experts (int): Number of experts in the MoE layer
        threads (int, optional): Number of threads per block. Defaults to 256.
    """

    def __init__(self, num_experts: int, threads: int = 256):
        self.func1 = moe_align_block_size_stage1(num_experts, threads)
        self.func2 = moe_align_block_size_stage2(num_experts, threads)

    def __call__(
        self,
        topk_ids: torch.Tensor,
        token_cnts: torch.Tensor,
        lock: torch.Tensor,
        expert_ids: torch.Tensor,
        sorted_topk_ids: torch.Tensor,
        total_tokens_per_expert: torch.Tensor,
        expert_start_index: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
    ):
        """
        Execute the two-stage MoE block size alignment process.
        
        Args:
            topk_ids (torch.Tensor): Input tensor containing expert assignments for each token
            token_cnts (torch.Tensor): Counter tensor for tracking tokens per expert
            lock (torch.Tensor): Synchronization lock for inter-block coordination
            expert_ids (torch.Tensor): Output tensor for expert IDs
            sorted_topk_ids (torch.Tensor): Output tensor for sorted token IDs
            total_tokens_per_expert (torch.Tensor): Output tensor for total tokens per expert
            expert_start_index (torch.Tensor): Output tensor for expert start indices
            num_tokens_post_pad (torch.Tensor): Output tensor for number of padded tokens
        """
        self.func1(topk_ids, token_cnts, lock, expert_ids,
                   total_tokens_per_expert, expert_start_index,
                   num_tokens_post_pad)
        self.func2(topk_ids, token_cnts, expert_start_index, sorted_topk_ids)


def moe_align_block_size_kernel(num_experts: int, threads: int = 256):
    """
    Factory function to create a MoEAlignBlockSize instance.
    
    Args:
        num_experts (int): Number of experts in the MoE layer
        threads (int, optional): Number of threads per block. Defaults to 256.
    
    Returns:
        MoEAlignBlockSize: An instance of the MoEAlignBlockSize class
    """
    return MoEAlignBlockSize(num_experts, threads)


def silu_and_mul_kernel(n: int, act_dtype: Union[str, DataType] = "float16"):
    """
    CUDA kernel for computing SiLU activation followed by element-wise multiplication.
    
    This kernel performs the operation: output = SiLU(input1) * input2
    where input1 and input2 are concatenated in the input tensor.
    
    Args:
        n (int): Size of the feature dimension
        act_dtype (Union[str, DataType], optional): Activation data type. Defaults to "float16".
    
    Returns:
        Compiled CUDA kernel function
    """
    m = symbol_var("total_tokens")
    adtype = data_type(act_dtype)
    threads = 256
    block_size = 2048

    with hidet.script_module() as script_module:

        @hidet.script
        def func(out_features: adtype[m, n], in_features: adtype[m, 2 * n]):
            # Step 1: Set up kernel parameters and thread indices
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = m * cdiv(n, block_size)
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            grid_n = cdiv(n, block_size)
            pid_n = pid % grid_n
            pid_m = pid // grid_n

            # Step 2: Set up masks for boundary handling
            mask_in1 = mask(auto_copy(), [i32(1), n - pid_n * block_size])
            mask_in2 = mask(auto_copy(), [i32(1), n - pid_n * block_size])
            mask_out = mask(auto_copy(), [i32(1), n - pid_n * block_size])

            # Step 3: Load input data
            g_in1 = tensor_view(in_features[pid_m, pid_n * block_size:],
                                TensorLayout((1, block_size), (2 * n, 1)),
                                "global")
            g_in2 = tensor_view(in_features[pid_m, pid_n * block_size + n:],
                                TensorLayout((1, block_size), (2 * n, 1)),
                                "global")
            r_in1 = make_tensor(adtype, layout_auto((1, block_size)),
                                "register")
            r_in2 = make_tensor(adtype, layout_auto((1, block_size)),
                                "register")
            txgin1 = partition_src(g_in1, auto_copy())
            txgin2 = partition_src(g_in2, auto_copy())
            txrin1 = partition_dst(r_in1, auto_copy())
            txrin2 = partition_dst(r_in2, auto_copy())
            copy(auto_copy((1, block_size)), txgin1, txrin1, mask_in1)
            copy(auto_copy((1, block_size)), txgin2, txrin2, mask_in2)

            # Step 4: Compute SiLU and multiplication
            r_out = make_tensor(adtype, layout_auto((1, block_size)),
                                "register")
            r_out = cast(silu(cast(r_in1, f32)) * cast(r_in2, f32), adtype)

            # Step 5: Store output
            g_out = tensor_view(out_features[pid_m, pid_n * block_size:],
                                TensorLayout((1, block_size), (n, 1)),
                                "global")
            txgout = partition_dst(g_out, auto_copy())
            txrout = partition_src(r_out, auto_copy())
            copy(auto_copy((1, block_size)), txrout, txgout, mask_out)

    func = script_module.build()
    return func


def moe_sum_kernel(experts_per_token: int,
                   k: int,
                   act_dtype: Union[str, DataType] = "float16"):
    """
    CUDA kernel for summing expert outputs in MoE.
    
    This kernel performs the weighted sum of expert outputs for each token.
    It aggregates the contributions from multiple experts based on their routing weights.
    
    Args:
        experts_per_token (int): Number of experts assigned to each token
        k (int): Size of the feature dimension
        act_dtype (Union[str, DataType], optional): Activation data type. Defaults to "float16".
    
    Returns:
        Compiled CUDA kernel function
    """
    m = symbol_var("num_tokens")
    adtype = data_type(act_dtype)
    threads = 256
    block_size = 2048

    with hidet.script_module() as script_module:

        @hidet.script
        def func(out_features: adtype[m, k],
                 in_features: adtype[m, experts_per_token, k]):
            # Step 1: Set up kernel parameters and thread indices
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = m * cdiv(k, block_size)
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            grid_n = cdiv(k, block_size)
            pid_n = pid % grid_n
            pid_m = pid // grid_n

            # Step 2: Initialize accumulator
            r_sum = make_tensor(f32, layout_auto((1, block_size)), "register")
            fill(r_sum, 0.0)

            # Step 3: Accumulate expert contributions
            for i in range(experts_per_token):
                mask_in = mask(auto_copy(), [i32(1), k - pid_n * block_size])
                g_in = tensor_view(in_features[pid_m, i, pid_n * block_size:],
                                   TensorLayout((1, block_size), (k, 1)),
                                   "global")
                r_in = make_tensor(adtype, layout_auto((1, block_size)),
                                   "register")
                txgin = partition_src(g_in, auto_copy())
                txrin = partition_dst(r_in, auto_copy())
                copy(auto_copy((1, block_size)), txgin, txrin, mask_in)
                r_sum = r_sum + cast(r_in, f32)

            # Step 4: Store final sum
            mask_out = mask(auto_copy(), [i32(1), k - pid_n * block_size])
            g_out = tensor_view(out_features[pid_m, pid_n * block_size:],
                                TensorLayout((1, block_size), (k, 1)),
                                "global")
            txgout = partition_dst(g_out, auto_copy())
            txrout = partition_src(cast(r_sum, adtype), auto_copy())
            copy(auto_copy((1, block_size)), txrout, txgout, mask_out)

    func = script_module.build()
    return func


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int,
                         num_experts: int):
    """
    Main function to perform MoE block size alignment.
    
    This function orchestrates the two-stage block size alignment process for MoE computation.
    It prepares the necessary tensors and executes the alignment kernels.
    
    Args:
        topk_ids (torch.Tensor): Input tensor containing expert assignments for each token
        block_size (int): Size of blocks for alignment
        num_experts (int): Number of experts in the MoE layer
    
    Returns:
        tuple: A tuple containing:
            - sorted_ids (torch.Tensor): Sorted token IDs aligned by expert
            - expert_ids (torch.Tensor): Expert IDs for each block
            - total_tokens_per_expert (torch.Tensor): Total tokens assigned to each expert
            - expert_start_index (torch.Tensor): Starting index for each expert's tokens
            - num_tokens_post_pad (torch.Tensor): Number of padded tokens after alignment
    """
    NUM_THREADS_PER_BLOCK = 64
    threads = NUM_THREADS_PER_BLOCK
    num_tokens = topk_ids.numel()
    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)
    from hidet.ffi import runtime_api

    runtime_api.set_symbol_value("total_tokens", num_tokens)
    runtime_api.set_symbol_value("block_size", block_size)
    func = moe_align_block_size_kernel(num_experts, threads)
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
    lock = torch.zeros((1, ), dtype=torch.int32, device=topk_ids.device)
    tokens_cnt = torch.zeros((cdiv(num_tokens, threads), num_experts),
                             dtype=torch.int32,
                             device=topk_ids.device)
    func(topk_ids, tokens_cnt, lock, expert_ids, sorted_ids,
         total_tokens_per_expert, expert_start_index, num_tokens_post_pad)
    return sorted_ids, expert_ids, total_tokens_per_expert, expert_start_index, num_tokens_post_pad


def cast_u4_to_f16(t: torch.Tensor,
                   act_dtype: Union[str, DataType] = "float16"):
    """
    Convert 4-bit unsigned integers to float16 values.
    
    This function performs the dequantization of 4-bit weights to float16 format.
    It's used for converting quantized weights back to their original precision.
    
    Args:
        t (torch.Tensor): Input tensor of 4-bit unsigned integers
        act_dtype (Union[str, DataType], optional): Target activation data type. Defaults to "float16".
    
    Returns:
        torch.Tensor: Dequantized tensor in float16 format
    """
    m, n = t.shape
    n = n * t.element_size() * 8 // u4.nbits
    ts = torch.empty(m, n, dtype=torch.float16, device="cuda")
    bm, bn = 64, 64
    threads = 128
    act_dtype = data_type(act_dtype)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: u4[m, n], wo: act_dtype[m, n]):
            # Step 1: Set up kernel parameters and thread indices
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = i64(pid // num_pid_n)
            pid_n = i64(pid % num_pid_n)

            # Step 2: Set up masks and load input data
            mski = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wi = tensor_view(wi[pid_m * bm:, pid_n * bn:],
                                TensorLayout((bm, bn), (n, 1)), "global")
            tr_wi = make_tensor(u4, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi, mski)

            # Step 3: Convert to target data type
            tr_w = cast(tr_wi, act_dtype)
            tr_wo = rearrange(tr_w, auto_layout, "register")

            # Step 4: Store output
            msko = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wo = tensor_view(wo[pid_m * bm:, pid_n * bn:],
                                TensorLayout((bm, bn), (n, 1)), "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo, msko)

    func = script_module.build()
    func(t, ts)
    return ts


def cast_f16_to_u4(t: torch.Tensor,
                   act_dtype: Union[str, DataType] = "float16"):
    """
    Convert float16 values to 4-bit unsigned integers.
    
    This function performs the quantization of float16 weights to 4-bit format.
    It's used for compressing weights to reduce memory usage and improve computation efficiency.
    
    Args:
        t (torch.Tensor): Input tensor in float16 format
        act_dtype (Union[str, DataType], optional): Source activation data type. Defaults to "float16".
    
    Returns:
        torch.Tensor: Quantized tensor in 4-bit unsigned integer format
    """
    m, n = t.shape
    ts = torch.empty(m, n // 2, dtype=torch.uint8, device="cuda")
    bm, bn = 64, 64
    threads = 128
    act_dtype = data_type(act_dtype)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: act_dtype[m, n], wo: u4[m, n]):
            # Step 1: Set up kernel parameters and thread indices
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = i64(pid // num_pid_n)
            pid_n = i64(pid % num_pid_n)

            # Step 2: Set up masks and load input data
            mski = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wi = tensor_view(wi[pid_m * bm:, pid_n * bn:],
                                TensorLayout((bm, bn), (n, 1)), "global")
            tr_wi = make_tensor(f16, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi, mski)

            # Step 3: Convert to 4-bit format
            tr_w = rearrange(tr_wi, auto_layout, "register")
            tr_wo = cast(tr_w, u4)

            # Step 4: Store output
            msko = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wo = tensor_view(wo[pid_m * bm:, pid_n * bn:],
                                TensorLayout((bm, bn), (n, 1)), "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo, msko)

    func = script_module.build()
    func(t, ts)
    return ts


def dqweight(qweight: torch.Tensor,
             act_dtype: Union[str, DataType] = "float16"):
    """
    Dequantize quantized weights back to float16 format.
    
    This function converts 1-bit, 2-bit, or 4-bit quantized weights back to their original float16 format.
    It's primarily used for sanity checking and debugging purposes.
    
    Args:
        qweight (torch.Tensor): Input tensor of quantized weights
        act_dtype (Union[str, DataType], optional): Target activation data type. Defaults to "float16".
    
    Returns:
        torch.Tensor: Dequantized weights in float16 format
    """
    weight_dtype = u4
    e, n, m = qweight.shape
    m = m // 16
    n = n * 16
    storage_dtype = i32
    pack_factor = storage_dtype.nbits // weight_dtype.nbits
    m = m * pack_factor
    out_dtype = data_type(act_dtype)
    if act_dtype == "float16":
        act_dtype = torch.float16
    else:
        act_dtype = torch.bfloat16
    w = torch.empty(e, m, n, dtype=act_dtype, device="cuda")

    bm = 64
    bn = 64
    threads = 128
    assert m % bm == 0 and n % bn == 0

    basic_block = TensorLayout(((8, 2), (2, 4, 2)), ((4, 2), (1, 64, 32)))
    #basic_block = TensorLayout(((8, 2), (2, 4, 2)), ((32, 4), (1, 8, 2)))
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

    def cvt(x):
        return out_dtype(x)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wq: weight_dtype[e, n, m], wdq: f16[e, m, n]):
            # Step 1: Set up kernel parameters and thread indices
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), e
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            bidy = i64(blockIdx.y)
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            # Step 2: Load quantized weights
            tg_wq = tensor_view(
                wq[bidy, pid_n * bn * n_stride:, pid_m * bm * m_stride:], gmem,
                "global")
            tr_wq = make_tensor(weight_dtype, auto_layout, "register")

            txgx_wq = partition_src(tg_wq, auto_copy())
            txrx_wq = partition_dst(tr_wq, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wq, txrx_wq)

            # Step 3: Convert to float16
            tr_w = cast(tr_wq, f16)
            tr_w_1 = rearrange(tr_w, auto_layout, "register")

            # Step 4: Store dequantized weights
            tg_w = tensor_view(
                wdq[bidy, pid_m * bm:(pid_m + 1) * bm,
                    pid_n * bn:(pid_n + 1) * bn],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )
            txgx_w = partition_dst(tg_w, auto_copy())
            txrx_w = partition_src(tr_w_1, auto_copy())

            copy(auto_copy((bm, bn)), txrx_w, txgx_w)

    func = script_module.build()
    func(qweight, w)
    return w


def weight_to_triton_weight(qweight,
                            scale,
                            zeros,
                            e,
                            k,
                            n,
                            group_size,
                            act_dtype: Union[str, DataType] = "float16"):
    """
    Convert quantized weights to Triton-compatible format.
    
    This function transforms the quantized weights and their associated scaling factors
    into a format that is compatible with Triton's MoE implementation.
    
    Args:
        qweight (torch.Tensor): Quantized weight tensor
        scale (torch.Tensor): Scaling factors for quantization
        zeros (torch.Tensor): Zero points for quantization
        e (int): Number of experts
        k (int): Input feature dimension
        n (int): Output feature dimension
        group_size (int): Size of groups for quantization
        act_dtype (Union[str, DataType], optional): Activation data type. Defaults to "float16".
    
    Returns:
        tuple: A tuple containing:
            - triton_qweight (torch.Tensor): Quantized weights in Triton format
            - triton_scale (torch.Tensor): Scaling factors in Triton format
            - triton_qzeros (torch.Tensor): Zero points in Triton format
    """
    triton_weight = dqweight(qweight).permute(0, 2, 1).reshape(-1, k)
    triton_qweight = cast_f16_to_u4(triton_weight,
                                    act_dtype).reshape(e, n, k // 2)
    triton_scale = scale.permute(0, 2, 1)
    triton_zeros = zeros.reshape(e, k // group_size, n // 2,
                                 2).permute(0, 2, 1,
                                            3).reshape(-1, k // group_size * 2)
    triton_qzeros = cast_f16_to_u4(triton_zeros,
                                   act_dtype).reshape(e, n // 2,
                                                      k // group_size)

    return triton_qweight, triton_scale, triton_qzeros