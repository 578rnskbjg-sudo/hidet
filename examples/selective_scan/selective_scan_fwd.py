from typing import Union

import torch
import hidet
import numpy as np

from hidet.ir.expr import Expr, symbol_var
from hidet.ir.dtypes import DataType, data_type
from hidet.lang.types import i32, f16, f32
from hidet.lang.cuda import blockIdx, threadIdx, cp_async_commit_group, cp_async_wait_group, syncthreads
from hidet.lang import attrs, grid

from hidet.ir.library import tune
from hidet.utils.py import cdiv

from hidet.ir.cute import layout_auto, auto_layout
from hidet.ir.cute.layout import TensorLayout
from hidet.ir.cute.algorithm import auto_copy
from hidet.ir.cute.ops import (cast, copy, fill, make_tensor, mask, mma,
                               partition_dst, partition_src, rearrange,
                               tensor_view, reduce_sum, exp2, exp)


LOG2E = np.log(2.0)


def data(d, n, total_length, input_dtype="float16", weight_dtype="float32", device="cuda"):
    input_dtype = getattr(torch, input_dtype)
    weight_dtype = getattr(torch, weight_dtype)
    u = torch.randint(low=-2, high=2, size=(total_length, d), dtype=input_dtype, device=device)
    delta = torch.randint(low=-2, high=2, size=(total_length, d), dtype=input_dtype, device=device)
    A = torch.randint(low=-2, high=2, size=(d, n), dtype=weight_dtype, device=device)
    B = torch.randint(low=-2, high=2, size=(total_length, n), dtype=input_dtype, device=device)
    C = torch.randint(low=-2, high=2, size=(total_length, n), dtype=input_dtype, device=device)
    D = torch.randint(low=-2, high=2, size=(d,), dtype=torch.float32, device=device)
    z = torch.randint(low=-2, high=2, size=(total_length, d), dtype=input_dtype, device=device)
    delta_bias = torch.randint(low=-2, high=2, size=(d,), dtype=torch.float32, device=device)
    return u, delta, A, B, C, D, z, delta_bias


class SelectiveScanFn:

    def __init__(self, dims: int, dstate: int, input_t: Union[str, DataType], weight_t: Union[str, DataType], delta_softplus: bool = True):
        self.dims = dims
        self.dstate = dstate
        self.input_t = data_type(input_t)
        self.weight_t = data_type(weight_t)
        self.delta_softplus = delta_softplus

    def __call__(self, u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.tensor, C: torch.Tensor, D: torch.Tensor, z: torch.Tensor, 
                 delta_bias: torch.Tensor, bias_softplus: torch.Tensor):
        pass

    @tune.space(2, block_b=[128, 256], block_d=[128, 256], block_l=[128, 256], block_n=[128, 256], sP=[2, 3])
    @tune.space(1, block_b=[128], block_d=[128], block_l=[128], block_n=[128], sP=[2])
    def modules(self, block_d: int, block_l: int, block_n: int, sP: int):
        return self.scan_fwd(block_d, block_l, block_n, sP)
    
    def scan_fwd(self, block_d: int, block_l: int, block_n: int, sP: int):
        batch_size = symbol_var("batch_size")
        total_length = symbol_var("total_length")
        d = self.dims 
        n = self.dstate
        input_t = self.input_t
        weight_t = self.weight_t

        blocks_b = batch_size
        blocks_d = cdiv(d, block_d)
        blocks_n = cdiv(n, block_n)
        MAX_SEQLEN = 16384
        padded_seqlen = block_l * cdiv(MAX_SEQLEN, block_l)

        with hidet.script_module() as script_module:
        
            @hidet.script
            def func(u: input_t[total_length, d],
                     delta: input_t[total_length, d],
                     A: weight_t[d, n],
                     B: input_t[total_length, n],
                     C: input_t[total_length, n],
                     D: f32[d],
                     z: input_t[total_length, d],
                     delta_bias: f32[d],
                     out: input_t[total_length, d],
                     query_start_loc: i32[batch_size + 1]):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = 128
                attrs.cuda.grid_dim = blocks_b * blocks_d
                attrs.cuda.dynamic_smem_bytes = 0

                bid = blockIdx.x
                batch_idx = bid // blocks_d
                bid_d = bid % blocks_d

                sequence_start_index = query_start_loc[batch_idx]
                seqlen = query_start_loc[batch_idx + 1] - sequence_start_index

                blocks_l = cdiv(seqlen, block_l)

                gU = tensor_view(u[sequence_start_index:, bid_d * block_d:(bid_d + 1) * block_d], 
                                 TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gDelta = tensor_view(delta[sequence_start_index:, bid_d * block_d:(bid_d + 1) * block_d], 
                                 TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gZ = tensor_view(z[sequence_start_index:, bid_d * block_d:(bid_d + 1) * block_d], 
                                 TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gOut = tensor_view(out[sequence_start_index:, bid_d * block_d:(bid_d + 1) * block_d], 
                                 TensorLayout((block_d, block_n, padded_seqlen), (1, 0, d)), "global")
                gDeltaBias = tensor_view(delta_bias[bid_d * block_d:(bid_d + 1) * block_d], 
                                 TensorLayout((block_d, block_n, block_l), (1, 0, 0)), "global")
                gD = tensor_view(D[bid_d * block_d:(bid_d + 1) * block_d], 
                                 TensorLayout((block_d, block_n, block_l), (1, 0, 0)), "global")
                tXgDeltaBias = partition_src(gDeltaBias, auto_copy())
                tXgD = partition_src(gD, auto_copy())
                rDeltaBias = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 0, 0)), "register")
                rD = make_tensor(f32, layout_auto((block_d, block_n, block_l), (1, 0, 0)), "register")
                tXrDeltaBias = partition_dst(rDeltaBias, auto_copy())
                tXrD = partition_dst(rD, auto_copy())

                rA = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 1, 0)), "register")
                rU = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")
                rDelta = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (1, 0, 1)), "register")
                rB = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (0, 1, 1)), "register")
                rC = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (0, 1, 1)), "register")
                rZ = make_tensor(input_t, layout_auto((block_d, block_n, block_l), (0, 1, 1)), "register")

                sDelta = make_tensor(input_t, layout_auto((block_d, block_n, block_l, sP), (1, 0, 1, 1)), "shared")
                sU = make_tensor(input_t, layout_auto((block_d, block_n, block_l, sP), (1, 0, 1, 1)), "shared")
                sB = make_tensor(input_t, layout_auto((block_d, block_n, block_l, sP), (0, 1, 1, 1)), "shared")
                sC = make_tensor(input_t, layout_auto((block_d, block_n, block_l, sP), (0, 1, 1, 1)), "shared")

                tXgOut = partition_dst(gOut, auto_copy())
                tXgZ = partition_src(gZ, auto_copy())
                tXrZ = partition_dst(rZ, auto_copy())

                tUsU = partition_dst(sU, auto_copy())
                tDsDelta = partition_dst(sDelta, auto_copy())
                tBsB = partition_dst(sB, auto_copy())
                tCsC = partition_dst(sC, auto_copy())

                tXsU = partition_dst(sU, auto_copy())
                tXsDelta = partition_dst(sDelta, auto_copy())
                tXsB = partition_dst(sB, auto_copy())
                tXsC = partition_dst(sC, auto_copy())

                tXrU = partition_dst(rU, auto_copy())
                tXrDelta = partition_dst(rDelta, auto_copy())
                tXrB = partition_dst(rB, auto_copy())
                tXrC = partition_dst(rC, auto_copy())

                copy(auto_copy((block_d, block_n, block_l)), tXgDeltaBias, tXrDeltaBias)
                copy(auto_copy((block_d, block_n, block_l)), tXgD, tXrD)

                # load the input tensors
                for i in range(blocks_n):
                    gA = tensor_view(A[bid_d * block_d:(bid_d + 1) * block_d, i * block_n:(i + 1) * block_n], 
                                     TensorLayout((block_d, block_n, block_l), (n, 1, 0)), "global")
                    gB = tensor_view(B[sequence_start_index:, i * block_n:(i + 1) * block_n], 
                                     TensorLayout((block_d, block_n, padded_seqlen), (0, 1, n)), "global")
                    gC = tensor_view(C[sequence_start_index:, i * block_n:(i + 1) * block_n], 
                                     TensorLayout((block_d, block_n, padded_seqlen), (0, 1, n)), "global")

                    tXgA = partition_src(gA, auto_copy())
                    tXrA = partition_dst(rA, auto_copy())
                    copy(auto_copy((block_d, block_n, block_l)), tXgA, tXrA)

                    tUgU = partition_src(gU, auto_copy())
                    tDgDelta = partition_src(gDelta, auto_copy())
                    tBgB = partition_src(gB, auto_copy())
                    tCgC = partition_src(gC, auto_copy())

                    tUsU = partition_dst(sU, auto_copy())
                    tDsDelta = partition_dst(sDelta, auto_copy())
                    tBsB = partition_dst(sB, auto_copy())
                    tCsC = partition_dst(sC, auto_copy())

                    smem_pipe_write = 0
                    smem_pipe_read = 0
                    
                    for j in range(sP - 1):
                        copy(auto_copy((block_d, block_n, block_l)), tUgU[:, :, :, j], tUsU[:, :, :, smem_pipe_write])
                        copy(auto_copy((block_d, block_n, block_l)), tDgDelta[:, :, :, j], tDsDelta[:, :, :, smem_pipe_write])
                        copy(auto_copy((block_d, block_n, block_l)), tBgB[:, :, :, j], tBsB[:, :, :, smem_pipe_write])
                        copy(auto_copy((block_d, block_n, block_l)), tCgC[:, :, :, j], tCsC[:, :, :, smem_pipe_write])
                        cp_async_commit_group()
                        smem_pipe_write += 1

                    for j in range(blocks_l):
                        cp_async_wait_group(allow_on_fly_groups=sP - 2)
                        syncthreads()
                        
                        copy(auto_copy((block_d, block_n, block_l)), tXgZ[:, :, :, j], tXrZ)
                        
                        if j + sP < block_l:
                            copy(auto_copy((block_d, block_n, block_l)), tUgU[:, :, :, j + sP], tUsU[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tDgDelta[:, :, :, j + sP], tDsDelta[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tBgB[:, :, :, j + sP], tBsB[:, :, :, smem_pipe_write])
                            copy(auto_copy((block_d, block_n, block_l)), tCgC[:, :, :, j + sP], tCsC[:, :, :, smem_pipe_write])
                            smem_pipe_write += 1
                            if smem_pipe_write == sP:
                                smem_pipe_write = 0
                        cp_async_commit_group()

                        copy(auto_copy((block_d, block_n, block_l)), tXsU[:, :, :, smem_pipe_read], tXrU)
                        copy(auto_copy((block_d, block_n, block_l)), tXsDelta[:, :, :, smem_pipe_read], tXrDelta)
                        copy(auto_copy((block_d, block_n, block_l)), tXsB[:, :, :, smem_pipe_read], tXrB) 
                        copy(auto_copy((block_d, block_n, block_l)), tXsC[:, :, :, smem_pipe_read], tXrC)
                        smem_pipe_read += 1
                        if smem_pipe_read == sP:
                            smem_pipe_read = 0

                        delta = rDelta + rDeltaBias
                        du = rD * rU
                        # delta_u = softplus(delta) * rU
                        delta_u = delta * rU
                        rALog2e = rA * LOG2E
                        theta0 = delta * rALog2e
                        theta1 = delta_u * rB

                        # (d, n, l)
                        # replace the sum with scan
                        scan = theta0 + theta1
                        scan1 = scan * rC
                        scan2 = du + reduce_sum(scan1, axis=1) # (d, n, l)

                        scan3 = scan2 * rZ / (1 + exp(-rZ))

                        tXrScan = partition_src(scan3, auto_copy())
                        copy(auto_copy((block_d, block_n, block_l)), tXrScan, tXgOut[:, :, :, j])


                        #scan = stack([tXrA, tXrB], axis=2)
                        #inclusive_scan(scan)

                        #res = unpack(scan, index=0)
                        #res = res * tXrC

                        #res = reduce_sum(res, axis=2)

        return script_module.build()


if __name__ == "__main__":
    batch_size = 50
    seqlen = 1321
    d = 5120
    n = 32
    input_t = "float16"
    weight_t = "float32"
    
    total_length = batch_size * seqlen
    u, delta, A, B, C, D, z, delta_bias = data(d, n, total_length, input_t, weight_t)

    fn = SelectiveScanFn(dims=d, dstate=n, input_t=input_t, weight_t=weight_t)
    block_d = 16
    block_l = 256
    block_n = 32
    sP = 2
    fn.scan_fwd(block_d, block_l, block_n, sP)