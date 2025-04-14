from typing import Union, Optional

import torch
import hidet
import numpy as np
import sys

from hidet.ffi import runtime_api
from hidet.ir.expr import Expr, symbol_var, deref
from hidet.ir.expr import cast as ir_cast
from hidet.ir.primitives import math

from hidet.ir.type import DataType, data_type
from hidet.lang.types import i32, f16, f32
from hidet.lang.cuda import blockIdx, threadIdx, cp_async_commit_group, cp_async_wait_group, syncthreads
from hidet.lang import attrs, grid
from hidet.lang import printf
from hidet.graph.frontend.torch.utils import dtype_to_torch

from hidet.ir.library import tune
from hidet.utils.py import cdiv

from hidet.ir.cute import layout_auto
from hidet.ir.cute.layout import TensorLayout, ThrValAtom, TiledTensorLayout
from hidet.ir.cute.algorithm import auto_copy
from hidet.ir.cute.ops import (
    cast,
    copy,
    fill,
    make_tensor,
    mask,
    partition_dst,
    partition_src,
    tensor_view,
    reduce_sum,
    softplus,
    silu,
    exp2,
    mbarrier_wait,
    mbarrier_arrive,
    make_mbarriers,
    pack,
    get,
    inclusive_scan,
)

from hidet.ir.cute.contexts import warp_groups_producer, warp_groups_consumer
from hidet.utils.benchmark import do_bench


LOG2E = np.log2(np.e)


def SSM_scan_op(a: Expr, b: Expr):
    ab0 = ir_cast(~a, ~f32)
    ab1 = ir_cast(~b, ~f32)
    ab0x = deref(ab0)
    ab0y = deref(ab0 + 1)
    ab1x = deref(ab1)
    ab1y = deref(ab1 + 1)
    return math.make_vector(ab1x * ab0x, ab1x * ab0y + ab1y)


def data(max_batch_size, d, n, total_length, input_dtype="float16", weight_dtype="float32", device="cuda"):
    input_dtype = getattr(torch, input_dtype)
    weight_dtype = getattr(torch, weight_dtype)
    u = torch.randint(low=-2, high=2, size=(total_length, d), dtype=input_dtype, device=device) / 32
    ssm_states = torch.randint(low=-2, high=2, size=(max_batch_size, d, n), dtype=input_dtype, device=device)
    delta = torch.randint(low=-2, high=0, size=(total_length, d), dtype=input_dtype, device=device)
    A = torch.randint(low=-2, high=0, size=(d, n), dtype=weight_dtype, device=device)
    B = torch.randint(low=-2, high=2, size=(total_length, n), dtype=input_dtype, device=device)
    C = torch.randint(low=-2, high=2, size=(total_length, n), dtype=input_dtype, device=device)
    D = torch.randint(low=-2, high=2, size=(d,), dtype=torch.float32, device=device)
    z = torch.randint(low=-2, high=2, size=(total_length, d), dtype=input_dtype, device=device)
    delta_bias = torch.randint(low=-2, high=0, size=(d,), dtype=torch.float32, device=device)
    return u, ssm_states, delta, A, B, C, D, z, delta_bias


class SelectiveScanFn:
    def __init__(
        self,
        max_batch_size: int,
        dims: int,
        dstate: int,
        input_t: Union[str, DataType],
        weight_t: Union[str, DataType],
        delta_softplus: bool = True,
        update_ssm_state: bool = False,
    ):
        self.dims = dims
        self.dstate = dstate
        self.input_t = data_type(input_t)
        self.weight_t = data_type(weight_t)
        self.delta_softplus = delta_softplus
        self.max_batch_size = max_batch_size
        self.update_ssm_state = update_ssm_state
        self.compiled_function = None
        self._compile()

    def __call__(
        self,
        u: torch.Tensor,
        ssm_states: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: torch.Tensor,
        delta_bias: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        out_z: Optional[torch.Tensor] = None,
    ):
        assert self.compiled_function is not None
        func = self.compiled_function
        total_length = u.shape[0]
        batch_size = query_start_loc.shape[0] - 1
        runtime_api.set_symbol_value("total_length", total_length)
        runtime_api.set_symbol_value("batch_size", batch_size)
        if out_z is None:
            out_z = torch.empty((total_length, self.dims), dtype=dtype_to_torch(self.input_t), device="cuda")
        func(u, ssm_states, delta, A, B, C, D, z, delta_bias, out_z, query_start_loc, cache_indices)
        return out_z

    def create_fake_tensors(self, total_length):
        input_dtype = dtype_to_torch(self.input_t)
        weight_dtype = dtype_to_torch(self.weight_t)
        u = torch.randint(low=-2, high=2, size=(total_length, self.dims), dtype=input_dtype, device="cuda") / 32
        ssm_states = torch.randint(low=-2, high=2, size=(self.max_batch_size, self.dims, self.dstate), dtype=input_dtype, device="cuda")
        delta = torch.randint(low=-2, high=0, size=(total_length, self.dims), dtype=input_dtype, device="cuda")
        A = torch.randint(low=-2, high=0, size=(self.dims, self.dstate), dtype=weight_dtype, device="cuda")
        B = torch.randint(low=-2, high=2, size=(total_length, self.dstate), dtype=input_dtype, device="cuda")
        C = torch.randint(low=-2, high=2, size=(total_length, self.dstate), dtype=input_dtype, device="cuda")
        D = torch.randint(low=-2, high=2, size=(self.dims,), dtype=torch.float32, device="cuda")
        z = torch.randint(low=-2, high=2, size=(total_length, self.dims), dtype=input_dtype, device="cuda")
        delta_bias = torch.randint(low=-2, high=0, size=(self.dims,), dtype=torch.float32, device="cuda")
        return u, ssm_states, delta, A, B, C, D, z, delta_bias

    def _compile(self):
        modules = tune.extract_ir_modules(self.modules)

        batch_size = 50
        seqlen = 1321
        total_length = seqlen * batch_size

        u, ssm_states, delta, A, B, C, D, z, delta_bias = self.create_fake_tensors(total_length)
        out_z = torch.empty((total_length, self.dims), dtype=dtype_to_torch(self.input_t), device="cuda")
        runtime_api.set_symbol_value("total_length", total_length)
        runtime_api.set_symbol_value("batch_size", batch_size)
        query_start_loc = torch.zeros((batch_size + 1), dtype=torch.int32, device="cuda")
        query_start_loc[0] = 0
        for i in range(batch_size):
            query_start_loc[i + 1] = query_start_loc[i] + seqlen
        cache_indices = torch.zeros((batch_size), dtype=torch.int32, device="cuda")
        for i in range(batch_size):
            cache_indices[i] = i

        min_time = sys.float_info.max
        min_config = None
        min_func = None
        for module in modules:
            kwargs = module._tuning_kwargs
            block_d = kwargs["block_d"]
            block_l = kwargs["block_l"]
            sP = kwargs["sP"]
            func = module.build()

            def fn():
                func(u, ssm_states, delta, A, B, C, D, z, delta_bias, out_z, query_start_loc, cache_indices)

            time = do_bench(fn, percentiles=None)
            if time < min_time:
                min_time = time
                min_func = func
                min_config = (block_d, block_l, sP)
        print(f"best config: {min_config}, time: {min_time} ms")
        self.compiled_function = min_func

    @tune.space(2, block_d=[128, 256], block_l=[4, 8], sP=[2, 3, 4, 5, 6, 7, 8])
    @tune.space(1, block_d=[128], block_l=[4], sP=[5])
    def modules(self, block_d: int, block_l: int, sP: int):
        tune.check(self.dims % block_d == 0)
        if sP == 1:
            return self.scan_fwd_single_buffer(block_d, block_l, sP)
        else:
            return self.scan_fwd_pipelined(block_d, block_l, sP)

    def scan_fwd_single_buffer(self, block_d: int, block_l: int, sP: int):
        batch_size = symbol_var("batch_size")
        total_length = symbol_var("total_length")
        d = self.dims
        n = self.dstate
        input_t = self.input_t
        weight_t = self.weight_t

        block_n = n
        blocks_b = batch_size
        blocks_d = cdiv(d, block_d)
        blocks_n = cdiv(n, block_n)
        MAX_SEQLEN = 16384
        padded_seqlen = block_l * cdiv(MAX_SEQLEN, block_l)
        delta_softplus = self.delta_softplus
        update_ssm_state = self.update_ssm_state

        num_threads = 128 if block_d == 128 else 256
        num_elements_per_thread = block_d * block_n * block_l // num_threads
        vector_size_d = 4 // input_t.nbytes
        vector_size_n = num_elements_per_thread // (vector_size_d * block_l)
        remaining_vector_d = block_d // vector_size_d
        remaining_vector_n = block_n // vector_size_n
        thread_value_layout = TensorLayout(
            ((remaining_vector_d, remaining_vector_n), (vector_size_d, vector_size_n, block_l)), ((vector_size_d, vector_size_n * block_d), (1, block_d, block_d * block_n))
        )

        tv_atom = ThrValAtom("thread_block", (block_d, block_n, block_l), thread_value_layout)
        tiled_layout = TiledTensorLayout(tv_atom)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                u: input_t[total_length, d],
                ssm_states: input_t[max_batch_size, d, n],
                delta: input_t[total_length, d],
                A: weight_t[d, n],
                B: input_t[total_length, n],
                C: input_t[total_length, n],
                D: f32[d],
                z: input_t[total_length, d],
                delta_bias: f32[d],
                out_z: input_t[total_length, d],
                query_start_loc: i32[batch_size + 1],
                cache_indices: i32[batch_size],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = num_threads
                attrs.cuda.grid_dim = blocks_b * blocks_d
                attrs.cuda.dynamic_smem_bytes = 0

                bid = blockIdx.x
                batch_idx = bid // blocks_d
                bid_d = bid % blocks_d

                sequence_start_index = query_start_loc[batch_idx]
                seqlen = query_start_loc[batch_idx + 1] - sequence_start_index
                cache_index = cache_indices[batch_idx]
                if cache_index == -1:
                    return
                blocks_l = cdiv(seqlen, block_l)

                gU = tensor_view(u[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gDelta = tensor_view(delta[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gZ = tensor_view(z[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                # gOut = tensor_view(
                #    out[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d],
                #    TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)),
                #    "global",
                # )
                gOutZ = tensor_view(out_z[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gDeltaBias = tensor_view(delta_bias[bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, block_l), (1, 0, 0)), "global")
                gD = tensor_view(D[bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, block_l), (1, 0, 0)), "global")
                tXgDeltaBias = partition_src(gDeltaBias, auto_copy())
                tXgD = partition_src(gD, auto_copy())
                rDeltaBias = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 0, 0)), "register")
                rD = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 0, 0)), "register")
                tXrDeltaBias = partition_dst(rDeltaBias, auto_copy())
                tXrD = partition_dst(rD, auto_copy())

                rA = make_tensor(weight_t, layout_auto((block_d, block_n, block_l), (1, 1, 0)), "register")
                rU = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")
                rDelta = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")
                rB = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (0, 1, 1)), "register")
                rC = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (0, 1, 1)), "register")
                rZ = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")

                sDelta = make_tensor(input_t, TensorLayout((block_d, block_n, block_l), (1, 0, block_d)), "shared")
                sU = make_tensor(input_t, TensorLayout((block_d, block_n, block_l), (1, 0, block_d)), "shared")
                sB = make_tensor(input_t, TensorLayout((block_d, block_n, block_l), (0, 1, block_n)), "shared")
                sC = make_tensor(input_t, TensorLayout((block_d, block_n, block_l), (0, 1, block_n)), "shared")
                sZ = make_tensor(input_t, TensorLayout((block_d, block_n, block_l), (1, 0, block_d)), "shared")

                # tXgOut = partition_dst(gOut, auto_copy())
                tXgOutZ = partition_dst(gOutZ, auto_copy())

                tUsU = partition_dst(sU, auto_copy())
                tZsZ = partition_dst(sZ, auto_copy())
                tDsDelta = partition_dst(sDelta, auto_copy())
                tBsB = partition_dst(sB, auto_copy())
                tCsC = partition_dst(sC, auto_copy())

                tXsU = partition_src(sU, auto_copy())
                tXsZ = partition_src(sZ, auto_copy())
                tXsDelta = partition_src(sDelta, auto_copy())
                tXsB = partition_src(sB, auto_copy())
                tXsC = partition_src(sC, auto_copy())

                tXrU = partition_dst(rU, auto_copy())
                tXrZ = partition_dst(rZ, auto_copy())
                tXrDelta = partition_dst(rDelta, auto_copy())
                tXrA = partition_dst(rA, auto_copy())
                tXrB = partition_dst(rB, auto_copy())
                tXrC = partition_dst(rC, auto_copy())

                copy(auto_copy((block_d, block_n, block_l)), tXgDeltaBias, tXrDeltaBias)
                copy(auto_copy((block_d, block_n, block_l)), tXgD, tXrD)

                residue = seqlen % block_l
                if residue == 0:
                    residue = block_l
                mask_u = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                mask_delta = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                mask_z = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                mask_b = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                mask_c = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                # mask_out = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                mask_out_z = mask(auto_copy(), [i32(block_d), i32(block_n), residue])

                # load the input tensors
                for i in range(blocks_n):
                    gA = tensor_view(A[bid_d * block_d : (bid_d + 1) * block_d, i * block_n : (i + 1) * block_n,], TensorLayout((block_d, block_n, block_l), (n, 1, 0)), "global")
                    gB = tensor_view(B[sequence_start_index:, i * block_n : (i + 1) * block_n], TensorLayout((block_d, block_n, padded_seqlen), (0, 1, n)), "global")
                    gC = tensor_view(C[sequence_start_index:, i * block_n : (i + 1) * block_n], TensorLayout((block_d, block_n, padded_seqlen), (0, 1, n)), "global")

                    tXgA = partition_src(gA, auto_copy())
                    copy(auto_copy((block_d, block_n, block_l)), tXgA, tXrA)
                    rA_LOG2E = tXrA * LOG2E

                    gSSMStates = tensor_view(
                        ssm_states[cache_index, bid_d * block_d : (bid_d + 1) * block_d, i * block_n : (i + 1) * block_n,],
                        TensorLayout((block_d, block_n, block_l), (n, 1, 0)),
                        "global",
                    )
                    rSSMStates = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 1, 0)), "register")
                    rOnes = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 1, 0)), "register")
                    tXgSSMStates = partition_src(gSSMStates, auto_copy())
                    tXrSSMStates = partition_dst(rSSMStates, auto_copy())
                    copy(auto_copy((block_d, block_n, block_l)), tXgSSMStates, tXrSSMStates)
                    fill(rOnes, 1.0)
                    rRunningPrefix = pack(rOnes, cast(tXrSSMStates, f32))

                    tUgU = partition_src(gU, auto_copy())
                    tZgZ = partition_src(gZ, auto_copy())
                    tDgDelta = partition_src(gDelta, auto_copy())
                    tBgB = partition_src(gB, auto_copy())
                    tCgC = partition_src(gC, auto_copy())

                    for j in range(blocks_l):
                        if j == blocks_l - 1:
                            copy(auto_copy((block_d, block_n, block_l)), tUgU[:, :, :, j], tUsU, mask_u)
                            copy(auto_copy((block_d, block_n, block_l)), tDgDelta[:, :, :, j], tDsDelta, mask_delta)
                            copy(auto_copy((block_d, block_n, block_l)), tBgB[:, :, :, j], tBsB, mask_b)
                            copy(auto_copy((block_d, block_n, block_l)), tCgC[:, :, :, j], tCsC, mask_c)
                            copy(auto_copy((block_d, block_n, block_l)), tZgZ[:, :, :, j], tZsZ, mask_z)
                        else:
                            copy(auto_copy((block_d, block_n, block_l)), tUgU[:, :, :, j], tUsU)
                            copy(auto_copy((block_d, block_n, block_l)), tDgDelta[:, :, :, j], tDsDelta)
                            copy(auto_copy((block_d, block_n, block_l)), tBgB[:, :, :, j], tBsB)
                            copy(auto_copy((block_d, block_n, block_l)), tCgC[:, :, :, j], tCsC)
                            copy(auto_copy((block_d, block_n, block_l)), tZgZ[:, :, :, j], tZsZ)
                        cp_async_commit_group()
                        cp_async_wait_group(0)
                        syncthreads()

                        copy(auto_copy((block_d, block_n, block_l)), tXsU, tXrU)
                        copy(auto_copy((block_d, block_n, block_l)), tXsDelta, tXrDelta)
                        copy(auto_copy((block_d, block_n, block_l)), tXsB, tXrB)
                        copy(auto_copy((block_d, block_n, block_l)), tXsC, tXrC)

                        # t1 = rDelta + rDeltaBias # (d, n, l):(1, 0, 1)
                        if delta_softplus:
                            t1 = softplus(tXrDelta + tXrDeltaBias)  # (d, n, l):(1, 0, 1)
                        else:
                            t1 = tXrDelta + tXrDeltaBias  # (d, n, l):(1, 0, 1)
                        theta0 = exp2(t1 * rA_LOG2E)  # (d, n, l)
                        delta_u = t1 * tXrU
                        du = tXrD * tXrU  # (d, n, l):(1, 0, 1)
                        theta1 = delta_u * tXrB  # (d, n, l)
                        # theta0 = t1 * rA_log2e # (d, n, l)
                        # (d, n, l)
                        scan0 = pack(theta0, theta1)

                        scan_result = inclusive_scan(scan0, axis=2, init=rRunningPrefix, scan_op=SSM_scan_op, layout=tiled_layout, update_init=True)
                        scan2 = get(scan_result, 1) * tXrC

                        yc = reduce_sum(scan2, axis=1)

                        copy(auto_copy((block_d, block_n, block_l)), tXsZ, tXrZ)

                        add = du + yc  # (d, n, l):(1, 0, 1)

                        # tXrOut = partition_src(cast(add, input_t), auto_copy())
                        # if j == blocks_l - 1:
                        #    copy(auto_copy((block_d, block_n, block_l)), tXrOut, tXgOut[:, :, :, j], mask_out)
                        # else:
                        #    copy(auto_copy((block_d, block_n, block_l)), tXrOut, tXgOut[:, :, :, j])

                        rOutZ = add * silu(cast(tXrZ, f32))
                        tXrOutZ = partition_src(cast(rOutZ, input_t), auto_copy())
                        if j == blocks_l - 1:
                            copy(auto_copy((block_d, block_n, block_l)), tXrOutZ, tXgOutZ[:, :, :, j], mask_out_z)
                        else:
                            copy(auto_copy((block_d, block_n, block_l)), tXrOutZ, tXgOutZ[:, :, :, j])
                        syncthreads()

                    if update_ssm_state:
                        # update the ssm states
                        tXrSSMStates = partition_src(cast(get(rRunningPrefix, 1), input_t), auto_copy())
                        tXgSSMStates_ = partition_dst(gSSMStates, auto_copy())
                        copy(auto_copy((block_d, block_n, block_l)), tXrSSMStates, tXgSSMStates_)

        return script_module

    def scan_fwd_pipelined(self, block_d: int, block_l: int, sP: int):
        batch_size = symbol_var("batch_size")
        total_length = symbol_var("total_length")
        d = self.dims
        n = self.dstate
        input_t = self.input_t
        weight_t = self.weight_t
        update_ssm_state = self.update_ssm_state

        block_n = n
        blocks_b = batch_size
        blocks_d = cdiv(d, block_d)
        blocks_n = cdiv(n, block_n)
        MAX_SEQLEN = 16384
        padded_seqlen = block_l * cdiv(MAX_SEQLEN, block_l)
        delta_softplus = self.delta_softplus

        num_threads = 128 if block_d == 128 else 256
        num_elements_per_thread = block_d * block_n * block_l // num_threads
        vector_size_d = 4 // input_t.nbytes
        vector_size_n = num_elements_per_thread // (vector_size_d * block_l)
        remaining_vector_d = block_d // vector_size_d
        remaining_vector_n = block_n // vector_size_n
        thread_value_layout = TensorLayout(
            ((remaining_vector_d, remaining_vector_n), (vector_size_d, vector_size_n, block_l)), ((vector_size_d, vector_size_n * block_d), (1, block_d, block_d * block_n))
        )

        tv_atom = ThrValAtom("thread_block", (block_d, block_n, block_l), thread_value_layout)
        tiled_layout = TiledTensorLayout(tv_atom)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                u: input_t[total_length, d],
                ssm_states: input_t[max_batch_size, d, n],
                delta: input_t[total_length, d],
                A: weight_t[d, n],
                B: input_t[total_length, n],
                C: input_t[total_length, n],
                D: f32[d],
                z: input_t[total_length, d],
                delta_bias: f32[d],
                out_z: input_t[total_length, d],
                query_start_loc: i32[batch_size + 1],
                cache_indices: i32[batch_size],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = num_threads
                attrs.cuda.grid_dim = blocks_b * blocks_d
                attrs.cuda.dynamic_smem_bytes = 0

                bid = blockIdx.x
                batch_idx = bid // blocks_d
                bid_d = bid % blocks_d

                sequence_start_index = query_start_loc[batch_idx]
                seqlen = query_start_loc[batch_idx + 1] - sequence_start_index
                cache_index = cache_indices[batch_idx]
                if cache_index == -1:
                    return
                blocks_l = cdiv(seqlen, block_l)

                gU = tensor_view(u[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gDelta = tensor_view(delta[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gZ = tensor_view(z[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                # gOut = tensor_view(
                #    out[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d],
                #    TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)),
                #    "global",
                # )
                gOutZ = tensor_view(out_z[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gDeltaBias = tensor_view(delta_bias[bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, block_l), (1, 0, 0)), "global")
                gD = tensor_view(D[bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, block_l), (1, 0, 0)), "global")
                tXgDeltaBias = partition_src(gDeltaBias, auto_copy())
                tXgD = partition_src(gD, auto_copy())
                rDeltaBias = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 0, 0)), "register")
                rD = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 0, 0)), "register")
                tXrDeltaBias = partition_dst(rDeltaBias, auto_copy())
                tXrD = partition_dst(rD, auto_copy())

                rA = make_tensor(weight_t, layout_auto((block_d, block_n, block_l), (1, 1, 0)), "register")
                rU = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")
                rDelta = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")
                rB = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (0, 1, 1)), "register")
                rC = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (0, 1, 1)), "register")
                rZ = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")

                sA = make_tensor(weight_t, TensorLayout((block_d, block_n, block_l), (1, block_d, 0)), "shared")
                sDelta = make_tensor(input_t, TensorLayout((block_d, block_n, block_l, sP), (1, 0, block_d, block_l * block_d)), "shared")
                sU = make_tensor(input_t, TensorLayout((block_d, block_n, block_l, sP), (1, 0, block_d, block_l * block_d)), "shared")
                sB = make_tensor(input_t, TensorLayout((block_d, block_n, block_l, sP), (0, 1, block_n, block_l * block_n)), "shared")
                sC = make_tensor(input_t, TensorLayout((block_d, block_n, block_l, sP), (0, 1, block_n, block_l * block_n)), "shared")
                sZ = make_tensor(input_t, TensorLayout((block_d, block_n, block_l, sP), (1, 0, block_d, block_l * block_d)), "shared")
                # tXgOut = partition_dst(gOut, auto_copy())
                tXgOutZ = partition_dst(gOutZ, auto_copy())

                residue = seqlen % block_l
                if residue == 0:
                    residue = block_l
                mask_u = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                mask_delta = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                mask_z = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                mask_b = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                mask_c = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                # mask_out = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                mask_out_z = mask(auto_copy(), [i32(block_d), i32(block_n), residue])

                tUsU = partition_dst(sU, auto_copy())
                tZsZ = partition_dst(sZ, auto_copy())
                tDsDelta = partition_dst(sDelta, auto_copy())
                tBsB = partition_dst(sB, auto_copy())
                tCsC = partition_dst(sC, auto_copy())
                tAsA = partition_dst(sA, auto_copy())

                tXsU = partition_src(sU, auto_copy())
                tXsZ = partition_src(sZ, auto_copy())
                tXsDelta = partition_src(sDelta, auto_copy())
                tXsB = partition_src(sB, auto_copy())
                tXsC = partition_src(sC, auto_copy())
                tXsA = partition_src(sA, auto_copy())

                tXrU = partition_dst(rU, auto_copy())
                tXrZ = partition_dst(rZ, auto_copy())
                tXrDelta = partition_dst(rDelta, auto_copy())
                tXrA = partition_dst(rA, auto_copy())
                tXrB = partition_dst(rB, auto_copy())
                tXrC = partition_dst(rC, auto_copy())

                copy(auto_copy((block_d, block_n, block_l)), tXgDeltaBias, tXrDeltaBias)
                copy(auto_copy((block_d, block_n, block_l)), tXgD, tXrD)

                # load the input tensors
                for i in range(blocks_n):
                    gA = tensor_view(A[bid_d * block_d : (bid_d + 1) * block_d, i * block_n : (i + 1) * block_n,], TensorLayout((block_d, block_n, block_l), (n, 1, 0)), "global")
                    gB = tensor_view(B[sequence_start_index:, i * block_n : (i + 1) * block_n], TensorLayout((block_d, block_n, padded_seqlen), (0, 1, n)), "global")
                    gC = tensor_view(C[sequence_start_index:, i * block_n : (i + 1) * block_n], TensorLayout((block_d, block_n, padded_seqlen), (0, 1, n)), "global")

                    tAgA = partition_src(gA, auto_copy())
                    copy(auto_copy((block_d, block_n, block_l)), tAgA, tAsA)
                    cp_async_commit_group()
                    cp_async_wait_group(0)

                    gSSMStates = tensor_view(
                        ssm_states[cache_index, bid_d * block_d : (bid_d + 1) * block_d, i * block_n : (i + 1) * block_n,],
                        TensorLayout((block_d, block_n, block_l), (n, 1, 0)),
                        "global",
                    )
                    rSSMStates = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 1, 0)), "register")
                    rOnes = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 1, 0)), "register")
                    tXgSSMStates = partition_src(gSSMStates, auto_copy())
                    tXrSSMStates = partition_dst(rSSMStates, auto_copy())
                    copy(auto_copy((block_d, block_n, block_l)), tXgSSMStates, tXrSSMStates)
                    fill(rOnes, 1.0)
                    rRunningPrefix = pack(rOnes, cast(tXrSSMStates, f32))

                    tUgU = partition_src(gU, auto_copy())
                    tZgZ = partition_src(gZ, auto_copy())
                    tDgDelta = partition_src(gDelta, auto_copy())
                    tBgB = partition_src(gB, auto_copy())
                    tCgC = partition_src(gC, auto_copy())

                    smem_pipe_write = 0
                    smem_pipe_read = 0

                    for j in range(sP - 1):
                        if j < blocks_l:
                            copy(auto_copy((block_d, block_n, block_l)), tUgU[:, :, :, j], tUsU[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tDgDelta[:, :, :, j], tDsDelta[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tBgB[:, :, :, j], tBsB[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tCgC[:, :, :, j], tCsC[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tZgZ[:, :, :, j], tZsZ[:, :, :, smem_pipe_write])
                        cp_async_commit_group()
                        smem_pipe_write += 1
                    cp_async_wait_group(allow_on_fly_groups=sP - 2)
                    syncthreads()

                    for j in range(blocks_l):
                        copy(auto_copy((block_d, block_n, block_l)), tXsA, tXrA)
                        rA_LOG2E = tXrA * LOG2E

                        if j + sP == blocks_l:
                            copy(auto_copy((block_d, block_n, block_l)), tUgU[:, :, :, j + sP - 1], tUsU[:, :, :, smem_pipe_write], mask_u)
                            copy(auto_copy((block_d, block_n, block_l)), tDgDelta[:, :, :, j + sP - 1], tDsDelta[:, :, :, smem_pipe_write], mask_delta)
                            copy(auto_copy((block_d, block_n, block_l)), tBgB[:, :, :, j + sP - 1], tBsB[:, :, :, smem_pipe_write], mask_b)
                            copy(auto_copy((block_d, block_n, block_l)), tCgC[:, :, :, j + sP - 1], tCsC[:, :, :, smem_pipe_write], mask_c)
                            copy(auto_copy((block_d, block_n, block_l)), tZgZ[:, :, :, j + sP - 1], tZsZ[:, :, :, smem_pipe_write], mask_z)
                        elif j + sP < blocks_l:
                            copy(auto_copy((block_d, block_n, block_l)), tUgU[:, :, :, j + sP - 1], tUsU[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tDgDelta[:, :, :, j + sP - 1], tDsDelta[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tBgB[:, :, :, j + sP - 1], tBsB[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tCgC[:, :, :, j + sP - 1], tCsC[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tZgZ[:, :, :, j + sP - 1], tZsZ[:, :, :, smem_pipe_write])

                            smem_pipe_write += 1
                            if smem_pipe_write == sP:
                                smem_pipe_write = 0
                        cp_async_commit_group()

                        copy(auto_copy((block_d, block_n, block_l)), tXsU[:, :, :, smem_pipe_read], tXrU)
                        copy(auto_copy((block_d, block_n, block_l)), tXsDelta[:, :, :, smem_pipe_read], tXrDelta)
                        copy(auto_copy((block_d, block_n, block_l)), tXsB[:, :, :, smem_pipe_read], tXrB)
                        copy(auto_copy((block_d, block_n, block_l)), tXsC[:, :, :, smem_pipe_read], tXrC)

                        # t1 = rDelta + rDeltaBias # (d, n, l):(1, 0, 1)
                        if delta_softplus:
                            t1 = softplus(tXrDelta + tXrDeltaBias)  # (d, n, l):(1, 0, 1)
                        else:
                            t1 = tXrDelta + tXrDeltaBias  # (d, n, l):(1, 0, 1)
                        delta_u = t1 * tXrU
                        theta1 = delta_u * tXrB  # (d, n, l)
                        # theta0 = t1 * rA_log2e # (d, n, l)
                        theta0 = exp2(t1 * rA_LOG2E)  # (d, n, l)
                        # (d, n, l)
                        scan0 = pack(theta0, theta1)

                        scan_result = inclusive_scan(scan0, axis=2, init=rRunningPrefix, scan_op=SSM_scan_op, layout=tiled_layout, update_init=True)
                        scan2 = get(scan_result, 1) * tXrC

                        yc = reduce_sum(scan2, axis=1)

                        copy(auto_copy((block_d, block_n, block_l)), tXsZ[:, :, :, smem_pipe_read], tXrZ)

                        du = tXrD * tXrU  # (d, n, l):(1, 0, 1)
                        add = du + yc  # (d, n, l):(1, 0, 1)

                        # tXrOut = partition_src(cast(add, input_t), auto_copy())
                        # if j == blocks_l - 1:
                        #    copy(auto_copy((block_d, block_n, block_l)), tXrOut, tXgOut[:, :, :, j], mask_out)
                        # else:
                        #    copy(auto_copy((block_d, block_n, block_l)), tXrOut, tXgOut[:, :, :, j])

                        rOutZ = add * silu(cast(tXrZ, f32))
                        tXrOutZ = partition_src(cast(rOutZ, input_t), auto_copy())
                        if j == blocks_l - 1:
                            copy(auto_copy((block_d, block_n, block_l)), tXrOutZ, tXgOutZ[:, :, :, j], mask_out_z)
                        else:
                            copy(auto_copy((block_d, block_n, block_l)), tXrOutZ, tXgOutZ[:, :, :, j])

                        smem_pipe_read += 1
                        if smem_pipe_read == sP:
                            smem_pipe_read = 0
                        cp_async_wait_group(allow_on_fly_groups=sP - 2)
                        syncthreads()

                    if update_ssm_state:
                        # update the ssm states
                        tXrSSMStates = partition_src(cast(get(rRunningPrefix, 1), input_t), auto_copy())
                        tXgSSMStates_ = partition_dst(gSSMStates, auto_copy())
                        copy(auto_copy((block_d, block_n, block_l)), tXrSSMStates, tXgSSMStates_)

        return script_module

    def scan_fwd_warpspecialized(self, block_d: int, block_l: int, sP: int):
        batch_size = symbol_var("batch_size")
        total_length = symbol_var("total_length")
        d = self.dims
        n = self.dstate
        input_t = self.input_t
        weight_t = self.weight_t

        delta_softplus = self.delta_softplus

        block_n = n
        blocks_b = batch_size
        blocks_d = cdiv(d, block_d)
        MAX_SEQLEN = 16384
        padded_seqlen = block_l * cdiv(MAX_SEQLEN, block_l)

        tma_copy_tx = (block_d * block_l * 3 + block_n * block_l * 2) * input_t.nbytes

        num_producer_threads = 128
        num_consumer_threads = 256
        num_elements_per_thread = block_d * block_n * block_l // num_consumer_threads
        vector_size_d = 4 // input_t.nbytes
        vector_size_n = num_elements_per_thread // (vector_size_d * block_l)
        remaining_vector_d = block_d // vector_size_d
        remaining_vector_n = block_n // vector_size_n
        thread_value_layout = TensorLayout(
            ((remaining_vector_d, remaining_vector_n), (vector_size_d, vector_size_n, block_l)), ((vector_size_d, vector_size_n * block_d), (1, block_d, block_d * block_n))
        )

        tv_atom = ThrValAtom("thread_block", (block_d, block_n, block_l), thread_value_layout)
        tiled_layout = TiledTensorLayout(tv_atom)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                u: input_t[total_length, d],
                ssm_states: input_t[max_batch_size, d, n],
                delta: input_t[total_length, d],
                A: weight_t[d, n],
                B: input_t[total_length, n],
                C: input_t[total_length, n],
                D: f32[d],
                z: input_t[total_length, d],
                delta_bias: f32[d],
                out_z: input_t[total_length, d],
                query_start_loc: i32[batch_size + 1],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = num_producer_threads + num_consumer_threads
                attrs.cuda.min_blocks = 1
                attrs.cuda.grid_dim = blocks_b * blocks_d
                attrs.cuda.dynamic_smem_bytes = 0

                bid = blockIdx.x
                batch_idx = bid // blocks_d
                bid_d = bid % blocks_d

                sequence_start_index = query_start_loc[batch_idx]
                seqlen = query_start_loc[batch_idx + 1] - sequence_start_index

                mbar_tma = make_mbarriers(sP)
                mbar_mma = make_mbarriers(sP)
                blocks_l = cdiv(seqlen, block_l)
                residue = seqlen % block_l
                if residue == 0:
                    residue = block_l

                gU = tensor_view(u[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gDelta = tensor_view(delta[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gZ = tensor_view(z[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")

                sDelta = make_tensor(input_t, TensorLayout((block_d, block_n, block_l, sP), (1, 0, block_d, block_l * block_d)), "shared")
                sU = make_tensor(input_t, TensorLayout((block_d, block_n, block_l, sP), (1, 0, block_d, block_l * block_d)), "shared")
                sB = make_tensor(input_t, TensorLayout((block_d, block_n, block_l, sP), (0, 1, block_n, block_l * block_n)), "shared")
                sC = make_tensor(input_t, TensorLayout((block_d, block_n, block_l, sP), (0, 1, block_n, block_l * block_n)), "shared")
                sZ = make_tensor(input_t, TensorLayout((block_d, block_n, block_l, sP), (1, 0, block_d, block_l * block_d)), "shared")

                syncthreads()

                with warp_groups_producer([2], 40):
                    smem_pipe_write = 0
                    write_phase = True

                    mask_u = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                    mask_delta = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                    mask_z = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                    mask_b = mask(auto_copy(), [i32(block_d), i32(block_n), residue])
                    mask_c = mask(auto_copy(), [i32(block_d), i32(block_n), residue])

                    i = 0
                    gB = tensor_view(B[sequence_start_index:, i * block_n : (i + 1) * block_n], TensorLayout((block_d, block_n, padded_seqlen), (0, 1, n)), "global")
                    gC = tensor_view(C[sequence_start_index:, i * block_n : (i + 1) * block_n], TensorLayout((block_d, block_n, padded_seqlen), (0, 1, n)), "global")

                    tUgU = partition_src(gU, auto_copy())
                    tZgZ = partition_src(gZ, auto_copy())
                    tDgDelta = partition_src(gDelta, auto_copy())
                    tBgB = partition_src(gB, auto_copy())
                    tCgC = partition_src(gC, auto_copy())

                    tUsU = partition_dst(sU, auto_copy())
                    tZsZ = partition_dst(sZ, auto_copy())
                    tDsDelta = partition_dst(sDelta, auto_copy())
                    tBsB = partition_dst(sB, auto_copy())
                    tCsC = partition_dst(sC, auto_copy())

                    for j in range(blocks_l):
                        if j >= sP:
                            mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)

                        if j == blocks_l - 1:
                            copy(auto_copy((block_d, block_n, block_l)), tUgU[:, :, :, j], tUsU[:, :, :, smem_pipe_write], mask_u, mbarrier=mbar_tma[smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tDgDelta[:, :, :, j], tDsDelta[:, :, :, smem_pipe_write], mask_delta, mbarrier=mbar_tma[smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tBgB[:, :, :, j], tBsB[:, :, :, smem_pipe_write], mask_b, mbarrier=mbar_tma[smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tCgC[:, :, :, j], tCsC[:, :, :, smem_pipe_write], mask_c, mbarrier=mbar_tma[smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tZgZ[:, :, :, j], tZsZ[:, :, :, smem_pipe_write], mask_z, mbarrier=mbar_tma[smem_pipe_write])
                        else:
                            copy(auto_copy((block_d, block_n, block_l)), tUgU[:, :, :, j], tUsU[:, :, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tDgDelta[:, :, :, j], tDsDelta[:, :, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tBgB[:, :, :, j], tBsB[:, :, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tCgC[:, :, :, j], tCsC[:, :, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tZgZ[:, :, :, j], tZsZ[:, :, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])

                        mbarrier_arrive(mbar_tma[smem_pipe_write], tma_copy_tx)
                        smem_pipe_write += 1
                        if smem_pipe_write == sP:
                            smem_pipe_write = 0
                            write_phase = not write_phase

                with warp_groups_consumer([0, 1], 232):
                    smem_pipe_read = 0
                    read_phase = False
                    smem_pipe_release = 0
                    release_phase = False

                    mask_out_z = mask(auto_copy(), [i32(block_d), i32(block_n), residue])

                    gOutZ = tensor_view(out_z[sequence_start_index:, bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                    gDeltaBias = tensor_view(delta_bias[bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, block_l), (1, 0, 0)), "global")
                    gD = tensor_view(D[bid_d * block_d : (bid_d + 1) * block_d], TensorLayout((block_d, block_n, block_l), (1, 0, 0)), "global")
                    tXgOutZ = partition_dst(gOutZ, auto_copy())
                    tXgDeltaBias = partition_src(gDeltaBias, auto_copy())
                    tXgD = partition_src(gD, auto_copy())

                    rDeltaBias = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 0, 0)), "register")
                    rD = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 0, 0)), "register")
                    tXrDeltaBias = partition_dst(rDeltaBias, auto_copy())
                    tXrD = partition_dst(rD, auto_copy())

                    rA = make_tensor(weight_t, layout_auto((block_d, block_n, block_l), (1, 1, 0)), "register")
                    rU = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")
                    rDelta = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")
                    rB = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (0, 1, 1)), "register")
                    rC = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (0, 1, 1)), "register")
                    rZ = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")

                    tXsU = partition_src(sU, auto_copy())
                    tXsZ = partition_src(sZ, auto_copy())
                    tXsDelta = partition_src(sDelta, auto_copy())
                    tXsB = partition_src(sB, auto_copy())
                    tXsC = partition_src(sC, auto_copy())

                    tXrU = partition_dst(rU, auto_copy())
                    tXrZ = partition_dst(rZ, auto_copy())
                    tXrDelta = partition_dst(rDelta, auto_copy())
                    tXrA = partition_dst(rA, auto_copy())
                    tXrB = partition_dst(rB, auto_copy())
                    tXrC = partition_dst(rC, auto_copy())

                    copy(auto_copy((block_d, block_n, block_l)), tXgDeltaBias, tXrDeltaBias)
                    copy(auto_copy((block_d, block_n, block_l)), tXgD, tXrD)

                    i = 0
                    gA = tensor_view(A[bid_d * block_d : (bid_d + 1) * block_d, i * block_n : (i + 1) * block_n,], TensorLayout((block_d, block_n, block_l), (n, 1, 0)), "global")

                    tXgA = partition_src(gA, auto_copy())
                    copy(auto_copy((block_d, block_n, block_l)), tXgA, tXrA)
                    rA_log2e = tXrA * LOG2E

                    gSSMStates = tensor_view(
                        ssm_states[batch_idx, bid_d * block_d : (bid_d + 1) * block_d, i * block_n : (i + 1) * block_n,],
                        TensorLayout((block_d, block_n, block_l), (n, 1, 0)),
                        "global",
                    )
                    rSSMStates = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 1, 0)), "register")
                    rOnes = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 1, 0)), "register")
                    tXgSSMStates = partition_src(gSSMStates, auto_copy())
                    tXrSSMStates = partition_dst(rSSMStates, auto_copy())
                    copy(auto_copy((block_d, block_n, block_l)), tXgSSMStates, tXrSSMStates)
                    fill(rOnes, 1.0)
                    rRunningPrefix = pack(rOnes, cast(tXrSSMStates, f32))

                    for j in range(blocks_l):
                        mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)

                        copy(auto_copy((block_d, block_n, block_l)), tXsU[:, :, :, smem_pipe_read], tXrU)
                        copy(auto_copy((block_d, block_n, block_l)), tXsDelta[:, :, :, smem_pipe_read], tXrDelta)
                        copy(auto_copy((block_d, block_n, block_l)), tXsB[:, :, :, smem_pipe_read], tXrB)
                        copy(auto_copy((block_d, block_n, block_l)), tXsC[:, :, :, smem_pipe_read], tXrC)

                        # t1 = rDelta + rDeltaBias # (d, n, l):(1, 0, 1)
                        if delta_softplus:
                            t1 = softplus(tXrDelta + tXrDeltaBias)  # (d, n, l):(1, 0, 1)
                        else:
                            t1 = tXrDelta + tXrDeltaBias
                        delta_u = t1 * tXrU
                        theta1 = delta_u * tXrB  # (d, n, l)
                        theta0 = exp2(t1 * rA_log2e)  # (d, n, l)
                        scan0 = pack(theta0, theta1)

                        scan_result = inclusive_scan(scan0, axis=2, init=rRunningPrefix, scan_op=SSM_scan_op, layout=tiled_layout, update_init=True)
                        scan2 = get(scan_result, 1) * tXrC

                        yc = reduce_sum(scan2, axis=1)

                        copy(auto_copy((block_d, block_n, block_l)), tXsZ[:, :, :, smem_pipe_read], tXrZ)
                        mbarrier_arrive(mbar_mma[smem_pipe_release])

                        du = tXrD * tXrU  # (d, n, l)
                        add = du + yc  # (d, n, l)

                        smem_pipe_read += 1
                        if smem_pipe_read == sP:
                            smem_pipe_read = 0
                            read_phase = not read_phase
                        smem_pipe_release += 1
                        if smem_pipe_release == sP:
                            smem_pipe_release = 0
                            release_phase = not release_phase

                        rOutZ = add * silu(cast(tXrZ, f32))
                        tXrOutZ = partition_dst(cast(rOutZ, input_t), auto_copy())
                        if j == blocks_l - 1:
                            copy(auto_copy((block_d, block_n, block_l)), tXrOutZ, tXgOutZ[:, :, :, j], mask_out_z)
                        else:
                            copy(auto_copy((block_d, block_n, block_l)), tXrOutZ, tXgOutZ[:, :, :, j])

        return script_module


def selective_scan_fn(
    max_batch_size: int, dims: int, dstate: int, input_t: Union[str, DataType], weight_t: Union[str, DataType], delta_softplus: bool = True, update_ssm_state: bool = False
):
    return SelectiveScanFn(max_batch_size, dims, dstate, input_t, weight_t, delta_softplus, update_ssm_state)


if __name__ == "__main__":
    hidet.option.cache_dir("./demo_selective_scan")
    hidet.option.debug_cache_tuning(True)
    hidet.option.save_lower_ir(True)
    hidet.option.search_space(2)
    hidet.option.num_local_workers(1)
    batch_size = 50
    seqlen = 1321
    d = 5120
    n = 32
    input_t = "float16"
    weight_t = "float32"
    max_batch_size = 256

    total_length = batch_size * seqlen
    u, ssm_states, delta, A, B, C, D, z, delta_bias = data(max_batch_size, d=d, n=n, total_length=total_length, input_dtype=input_t, weight_dtype=weight_t)
    query_start_loc = torch.zeros((batch_size + 1), dtype=torch.int32, device="cuda")
    query_start_loc[0] = 0
    for i in range(batch_size):
        query_start_loc[i + 1] = query_start_loc[i] + seqlen
    print(query_start_loc)
    cache_indices = torch.zeros((batch_size), dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        cache_indices[i] = i

    fn = SelectiveScanFn(max_batch_size=max_batch_size, dims=d, dstate=n, input_t=input_t, weight_t=weight_t)
    out_z = torch.empty((total_length, d), dtype=torch.float16, device="cuda")
    fn(u, ssm_states, delta, A, B, C, D, z, delta_bias, query_start_loc, cache_indices, out_z)

    def fn1():
        return fn(u, ssm_states, delta, A, B, C, D, z, delta_bias, query_start_loc, cache_indices, out_z)

    time = do_bench(fn1, percentiles=None)
    print(f"selective scan time(fn): {time} ms")

    block_d = 128
    block_l = 4
    block_n = 32
    sP = 5
    func_pipelined = fn.scan_fwd_pipelined(block_d, block_l, sP).build()
    func_single_buffer = fn.scan_fwd_single_buffer(block_d, block_l, sP).build()
    block_d = 256
    block_l = 4
    block_n = 32
    sP = 15
    func_warp_specialized = fn.scan_fwd_warpspecialized(block_d, block_l, sP).build()

    has_initial_state = torch.ones((batch_size,), dtype=torch.int32, device="cuda")
    has_initial_state = has_initial_state.to(torch.bool)
    print(has_initial_state)

    runtime_api.set_symbol_value("batch_size", batch_size)
    runtime_api.set_symbol_value("total_length", total_length)
    out_z_pipelined = torch.randn((total_length, d), dtype=torch.float16, device="cuda")
    out_z_single_buffer = torch.randn((total_length, d), dtype=torch.float16, device="cuda")
    out_z_warp_specialized = torch.randn((total_length, d), dtype=torch.float16, device="cuda")
    func = func_warp_specialized

    func(u, ssm_states, delta, A, B, C, D, z, delta_bias, out_z_warp_specialized, query_start_loc)
    print(out_z_warp_specialized)

    def fn8():
        func(u, ssm_states, delta, A, B, C, D, z, delta_bias, out_z_warp_specialized, query_start_loc)

    import cupy

    cupy.cuda.profiler.start()
    time = do_bench(fn8, percentiles=None)
    cupy.cuda.profiler.stop()
    memory_total = total_length * d * 4 * torch.float16.itemsize + total_length * n * 2 * torch.float16.itemsize + d * n * torch.float32.itemsize + d * 2 * torch.float32.itemsize
    print(f"selective scan time (warp specialized): {time} ms, memory bandwidth: {memory_total / time / 1e6} GB/s")

    func = func_pipelined

    func(u, ssm_states, delta, A, B, C, D, z, delta_bias, out_z_pipelined, query_start_loc, cache_indices)
    print(out_z_pipelined)

    def fn7():
        func(u, ssm_states, delta, A, B, C, D, z, delta_bias, out_z_pipelined, query_start_loc, cache_indices)

    import cupy

    cupy.cuda.profiler.start()
    time = do_bench(fn7, percentiles=None)
    cupy.cuda.profiler.stop()
    memory_total = total_length * d * 4 * torch.float16.itemsize + total_length * n * 2 * torch.float16.itemsize + d * n * torch.float32.itemsize + d * 2 * torch.float32.itemsize
    print(f"selective scan time (pipelined): {time} ms, memory bandwidth: {memory_total / time / 1e6} GB/s")

    func = func_single_buffer
    func(u, ssm_states, delta, A, B, C, D, z, delta_bias, out_z_single_buffer, query_start_loc, cache_indices)
    print(out_z_single_buffer)

    def fn():
        func(u, ssm_states, delta, A, B, C, D, z, delta_bias, out_z_single_buffer, query_start_loc, cache_indices)

    cupy.cuda.profiler.start()
    time = do_bench(fn, percentiles=None)
    cupy.cuda.profiler.stop()
    memory_total = total_length * d * 4 * torch.float16.itemsize + total_length * n * 2 * torch.float16.itemsize + d * n * torch.float32.itemsize + d * 2 * torch.float32.itemsize
    print(f"selective scan time(single buffer): {time} ms, memory bandwidth: {memory_total / time / 1e6} GB/s")

    def fn1():
        u1 = u.transpose(0, 1).contiguous()

    time = do_bench(fn1, percentiles=None)
    print(f"transpose u time: {time} ms")
    memory_total = total_length * d * torch.float16.itemsize * 2
    print(f"transpose u time: {time} ms, memory bandwidth: {memory_total / time / 1e6} GB/s")

    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_scan_fn

    u1 = u.transpose(0, 1)
    delta1 = delta.transpose(0, 1)
    B1 = B.transpose(0, 1)
    C1 = C.transpose(0, 1)
    z1 = z.transpose(0, 1).contiguous()

    def fn2():
        return selective_scan_fn(u1, ssm_states, delta1, A, B1, C1, D, z1, delta_bias, True, query_start_loc, None, has_initial_state)

    out2 = fn2()
    out2 = out2.transpose(0, 1)
    np.testing.assert_allclose(out_z_pipelined.to(torch.float32).cpu(), out2.to(torch.float32).cpu(), rtol=1e-2, atol=1)
    np.testing.assert_allclose(out_z_single_buffer.to(torch.float32).cpu(), out2.to(torch.float32).cpu(), rtol=1e-2, atol=1)

    time = do_bench(fn2, percentiles=None)
    memory_total = total_length * d * 4 * torch.float16.itemsize + total_length * n * 2 * torch.float16.itemsize + d * n * torch.float32.itemsize + d * 2 * torch.float32.itemsize
    print(f"selective scan time (vLLM): {time} ms, memory bandwidth: {memory_total / time / 1e6} GB/s")
