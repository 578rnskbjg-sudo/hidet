from typing import List
from hidet.ir.cute.contexts import warp_groups_consumer, warp_groups_producer
import torch
import pytest

from hidet.lang.types import f8e4m3, f8e5m2, bf16, f16, f32, i32
from hidet.lang import attrs, grid
from hidet.lang.cuda import blockIdx, threadIdx
from hidet.ir.primitives.cuda.wgmma import wgmma_fence, wgmma_commit_group, wgmma_wait_group
from hidet.lang.cuda import syncthreads, cp_async_commit_group, cp_async_wait_group
from hidet.lang.constructs.declare import as_tensor_pointer
from hidet.utils.py import cdiv

import hidet
from hidet.ir.cute import canonicalize_thread_value_layout
from hidet.ir.cute.algorithm import MmaAtom, TiledMma, auto_copy
from hidet.ir.cute.layout import TensorLayout, Level
from hidet.ir.cute import layout_auto, auto_layout, product_each, right_inverse
from hidet.ir.cute.algorithm import CopyAtom, TiledCopy

from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    partition_A,
    partition_B,
    copy,
    mma,
    rearrange,
    cast,
    fill,
    make_mbarriers,
    mbarrier_arrive,
    mbarrier_try_wait,
    mbarrier_wait,
    wgmma_fence_operand,
    transpose,
    mask,
)
from hidet.utils.benchmark import do_bench
from hidet.utils import initialize
from hidet.ir.library import tune
from quant_utils import bench


_tiled_mma_lists: List[TiledMma] = []


@initialize()
def register_tiled_mma():
    for warpgroup_m in [1, 2]:
        for n in [32, 64, 96, 128] + list(range(192, 257, 16)):
            a = TensorLayout(((128,), (n, 16)), ((0,), (1, n)))
            b = TensorLayout(((128,), (64, 16)), ((0,), (1, 64)))
            c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((2, n, 16 * n), (1, 8 * n, 8)))
            mma_atom = MmaAtom("warp_group", (n, 64, 16), a, b, c, c)
            wg_in_threadblock = Level("warp_group", "thread_block", (1, warpgroup_m), TensorLayout((1, warpgroup_m)), (1, 1))
            tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])
            _tiled_mma_lists.append(tiled_mma)
            #a = TensorLayout(((128,), (64, 16)), ((0,), (1, 64)))
            #b = TensorLayout(((128,), (n, 16)), ((0,), (1, n)))
            #c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((128, 1, 16), (64, 8, 512)))
            #mma_atom = MmaAtom("warp_group", (64, n, 16), a, b, c, c)
            #wg_in_threadblock = Level("warp_group", "thread_block", (warpgroup_m, 1), TensorLayout((warpgroup_m, 1)), (1, 1))
            #tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])
            #_tiled_mma_lists.append(tiled_mma)


class WarpSpecializedGemmSplit:
    def __init__(self, m, n, k, qdim, kdim, vdim, zdim):
        self.m = m
        self.n = n
        self.k = k
        self.qdim = qdim
        self.kdim = kdim
        self.vdim = vdim
        self.zdim = zdim
        assert n % (qdim + kdim + vdim + zdim) == 0

    def modules(self):
        return tune.extract_ir_modules(self._candidates)

    def build(self):
        from hashlib import sha256
        from hidet.drivers import build_ir_module
        from hidet.runtime import load_compiled_module
        from tqdm import tqdm
        from hidet.utils.multiprocess import parallel_imap_2ndlevel

        def build_job(args):
            ir_module, output_dir = args
            build_ir_module(ir_module, output_dir, target='cuda')

        ir_modules = self.modules()
        output_dirs = []
        for mod in ir_modules:
            hash_dir = sha256(str(mod).encode()).hexdigest()[:16]
            output_dir = hidet.utils.cache_dir('ir_modules', hash_dir)
            output_dirs.append(output_dir)

        jobs = [(ir_module, output_dir) for ir_module, output_dir in zip(ir_modules, output_dirs)]

        for _ in tqdm(parallel_imap_2ndlevel(build_job, jobs, is_remote_allowed=True), desc="Compiling", total=len(jobs), ncols=80):
            pass
        funcs = [load_compiled_module(output_dir) for output_dir in output_dirs]
        return list(zip(ir_modules, funcs))

    @tune.space(2, cluster_m=[1, 2, 4, 8, 16], cluster_n=[1, 2, 4, 8, 16], tiled_mma=_tiled_mma_lists, k_pipe_max=[4, 5, 6, -1], bk=[64])
    def _candidates(self, cluster_m, cluster_n, tiled_mma: TiledMma, k_pipe_max, bk):
        m, n, k = self.m, self.n, self.k
        qdim, kdim, vdim, zdim = self.qdim, self.kdim, self.vdim, self.zdim
        p = n // (qdim + kdim + vdim + zdim)
        qkv = (qdim + kdim + vdim) * p
        z = zdim * p
        a_shape, a_tv = tiled_mma.a_tv_layout()
        b_shape, b_tv = tiled_mma.b_tv_layout()
        c_shape, c_tv = tiled_mma.c_tv_layout()

        a_t, _ = canonicalize_thread_value_layout(a_tv)
        _, _ = canonicalize_thread_value_layout(b_tv)
        _, _ = canonicalize_thread_value_layout(c_tv)

        bm, inst_k = a_shape
        bn, inst_k_ = b_shape
        bm_, bn_ = c_shape
        assert bm == bm_ and bn == bn_ and inst_k == inst_k_
        threads = a_t.size()
        tma_copy_tx = (bm * bk + bn * bk) * f16.nbytes
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        sm_ver = hidet.option.cuda.get_arch_pair()
        sm_ver = sm_ver[0] * 10 + sm_ver[1]
        smem_limit = smem_limits[sm_ver]
        if k_pipe_max == -1:
            k_pipe_max = int(smem_limit / tma_copy_tx)
        k_pipe_mma = 1
        tune.check(tma_copy_tx * k_pipe_max <= smem_limit)
        device_prop = hidet.cuda.properties()
        num_sms = device_prop.multiProcessorCount
        # tune.check((m < 128 and bm == 64) or (m >= 128 and ((m % 128 == 0 and bm == 128) or (m % 128 != 0))))

        num_consumer_threads = threads
        num_producer_threads = 128

        if num_consumer_threads == 256:
            producer_warpgroups = [2]
            consumer_warpgroups = [0, 1]
        elif num_consumer_threads == 128:
            producer_warpgroups = [1]
            consumer_warpgroups = [0]

        # This is a tunable knob for cluster layout
        cluster_layout = TensorLayout((cluster_m, cluster_n), (1, cluster_m))
        cluster_id2mn = right_inverse(cluster_layout)
        cluster_size = cluster_layout.size()
        tune.check(cluster_size <= 16)
        cluster_shape = product_each(cluster_layout.shape_tuple)
        cluster_m, cluster_n = cluster_shape
        grid_size = cluster_size * cdiv(m, cluster_m * bm) * cdiv(n, cluster_n * bn)
        cta_ms = cdiv(m, cluster_m * bm) * cluster_m
        cta_ns = cdiv(n, cluster_n * bn) * cluster_n
        min_cta_dim = min(cta_ms, cta_ns)
        if min_cta_dim >= 6:
            log_swizzle_size = 3
        elif min_cta_dim >= 3:
            log_swizzle_size = 2
        elif min_cta_dim >= 2:
            log_swizzle_size = 1
        else:
            log_swizzle_size = 0
        cluster_blk_major = cdiv(n, cluster_n * bn)
        tune.check(num_sms % cluster_size == 0)
        unroll = f"u{k_pipe_max}"
        tune.check(qdim % bn == 0)
        tune.check(kdim % bn == 0)
        tune.check(vdim % bn == 0)
        tune.check(zdim % bn == 0)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(a: f16[m, k], b_ptr: ~f16, qkvt: f16[m, qkv], zt: f16[m, z]):
                # Kernel Configuration
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = num_producer_threads + num_consumer_threads  # 12 warps total
                attrs.cuda.grid_dim = grid_size, 1, 1  # Grid dimensions based on matrix size
                attrs.cuda.cluster_dim = cluster_size
                attrs.cuda.min_blocks = 1
                attrs.cuda.dynamic_smem_bytes = 0  # No dynamic shared memory required

                pid = blockIdx.x
                # Block indices for grid-level parallelism
                cluster_id = pid % cluster_size
                cluster_mn = cluster_id2mn(cluster_id)
                cluster_index_m = cluster_mn // cluster_n
                cluster_index_n = cluster_mn % cluster_n
                grid_id = pid // cluster_size
                offset = grid_id & ((1 << log_swizzle_size) - 1)
                extra = grid_id >> log_swizzle_size
                cluster_idx_minor_div_swizzle = extra // cluster_blk_major
                cluster_idx_major = extra % cluster_blk_major
                cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset
                bid_x = cluster_idx_minor * cluster_m + cluster_index_m
                bid_y = cluster_idx_major * cluster_n + cluster_index_n

                # Initialize memory barriers for producer-consumer synchronization
                mbar_tma = make_mbarriers(k_pipe_max)  # For TMA operations
                mbar_mma = make_mbarriers(k_pipe_max)  # For MMA operations

                # Set up tensor views for global memory access
                # Matrix A: Global memory view with strided layout
                tg_a = tensor_view(a, TensorLayout((m, k), (k, 1)), "global", (bm, k), (bid_x * bm, 0))
                # Matrix B: Global memory view with strided layout
                b = as_tensor_pointer(b_ptr, "float16", [n, k])
                tg_b = tensor_view(b, TensorLayout((n, k), (k, 1)), "global", (bn, k), (bid_y * bn, 0))

                # Allocate shared memory tensors with pipelined layout
                ts_b = make_tensor("float16", layout_auto((bn, bk, k_pipe_max)), "shared")
                ts_a = make_tensor("float16", layout_auto((bm, bk, k_pipe_max)), "shared")

                # Producer Warp Group: Responsible for data movement from global to shared memory
                with warp_groups_producer(producer_warpgroups, num_regs=24):
                    # Pipeline control variables
                    smem_pipe_write = 0
                    write_phase = True

                    # Set up tensor partitions for efficient data movement
                    txga = partition_src(tg_a, auto_copy())
                    txsa = partition_dst(ts_a, auto_copy())
                    txgb = partition_src(tg_b, auto_copy())
                    txsb = partition_dst(ts_b, auto_copy())

                    # Main producer loop: Move data from global to shared memory
                    k_blocks = cdiv(k, bk)
                    for ko in grid(k_blocks, attrs=unroll):
                        # Wait for previous MMA operations to complete if pipeline is full
                        if ko >= k_pipe_max:
                            mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)

                        # Copy data from global to shared memory using TMA
                        copy(auto_copy((bm, bk)), txga[:, :, ko], txsa[:, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])
                        copy(auto_copy((bn, bk)), txgb[:, :, ko], txsb[:, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])

                        # Signal completion of TMA operation
                        mbarrier_arrive(mbar_tma[smem_pipe_write], tma_copy_tx)

                        # Update pipeline stage
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

                # Consumer Warp Group: Responsible for matrix multiplication computation
                with warp_groups_consumer(consumer_warpgroups, num_regs=240):
                    # Pipeline control variables
                    smem_pipe_read = 0
                    read_phase = False
                    smem_pipe_release = 0
                    release_phase = False

                    # Allocate register tensors for computation
                    tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
                    fill(tr_c, 0.0)  # Initialize accumulation register

                    # Set up tensor partitions for computation
                    txSa = partition_A(ts_a, tiled_mma)
                    txSb = partition_B(ts_b, tiled_mma)

                    # Main computation loop
                    k_blocks = cdiv(k, bk)
                    k_tiles = cdiv(bk, inst_k)

                    wgmma_fence_operand(tr_c)
                    for ko in grid(k_pipe_mma, attrs=unroll):
                        mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                        wgmma_fence()
                        for ki in grid(k_tiles, attrs="u+"):
                            mma(tiled_mma, tr_c, txSa[:, :, ki, smem_pipe_read], txSb[:, :, ki, smem_pipe_read], tr_c, cluster_layout=cluster_layout)
                        wgmma_commit_group()
                        smem_pipe_read += 1
                        if smem_pipe_read == k_pipe_max:
                            smem_pipe_read = 0
                            read_phase = not read_phase
                    wgmma_fence_operand(tr_c)

                    for ko in grid(k_blocks - k_pipe_mma, attrs=unroll):
                        mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                        wgmma_fence_operand(tr_c)
                        wgmma_fence()
                        for ki in grid(k_tiles, attrs="u+"):
                            mma(tiled_mma, tr_c, txSa[:, :, ki, smem_pipe_read], txSb[:, :, ki, smem_pipe_read], tr_c)
                        wgmma_commit_group()

                        wgmma_wait_group(k_pipe_mma)
                        wgmma_fence_operand(tr_c)
                        mbarrier_arrive(mbar_mma[smem_pipe_release])
                        smem_pipe_release += 1
                        if smem_pipe_release == k_pipe_max:
                            smem_pipe_release = 0
                            release_phase = not release_phase
                        smem_pipe_read += 1
                        if smem_pipe_read == k_pipe_max:
                            smem_pipe_read = 0
                            read_phase = not read_phase

                    wgmma_wait_group(0)
                    for ko in range(k_pipe_mma):
                        mbarrier_arrive(mbar_mma[smem_pipe_release])
                        smem_pipe_release += 1
                        if smem_pipe_release == k_pipe_max:
                            smem_pipe_release = 0
                            release_phase = not release_phase
                    # Convert result to FP16 and prepare for global memory write
                    tr_C = rearrange(cast(tr_c, f16), auto_layout, "register")

                    BN = bid_y * bn
                    qkvz_dim = qdim + kdim + vdim + zdim
                    # Write result back to global memory
                    part = BN // qkvz_dim
                    remain = BN % qkvz_dim
                    if remain < qdim:
                        start = part * qdim + remain
                        tg_c = tensor_view(qkvt[bid_x * bm : (bid_x + 1) * bm, start : start + bn], TensorLayout((bm, bn), (qkv, 1)), "global")
                        txgc = partition_src(tg_c, auto_copy())
                        txrc = partition_dst(tr_C, auto_copy())
                        mask_c = mask(auto_copy(()), [m - bid_x * bm, i32(bn)])
                        copy(auto_copy((bm, bn)), txrc, txgc, mask_c)
                    elif remain < qdim + kdim:
                        start = p * qdim + part * kdim + remain - qdim
                        tg_c = tensor_view(qkvt[bid_x * bm : (bid_x + 1) * bm, start : start + bn], TensorLayout((bm, bn), (qkv, 1)), "global")
                        txgc = partition_src(tg_c, auto_copy())
                        txrc = partition_dst(tr_C, auto_copy())
                        mask_c = mask(auto_copy(()), [m - bid_x * bm, i32(bn)])
                        copy(auto_copy((bm, bn)), txrc, txgc, mask_c)
                    elif remain < qdim + kdim + vdim:
                        start = p * (qdim + kdim) + part * vdim + remain - (qdim + kdim)
                        tg_c = tensor_view(qkvt[bid_x * bm : (bid_x + 1) * bm, start : start + bn], TensorLayout((bm, bn), (qkv, 1)), "global")
                        txgc = partition_src(tg_c, auto_copy())
                        txrc = partition_dst(tr_C, auto_copy())
                        mask_c = mask(auto_copy(()), [m - bid_x * bm, i32(bn)])
                        copy(auto_copy((bm, bn)), txrc, txgc, mask_c)
                    else:
                        start = part * zdim + remain - (qdim + kdim + vdim)
                        tg_c = tensor_view(zt[bid_x * bm : (bid_x + 1) * bm, start : start + bn], TensorLayout((bm, bn), (z, 1)), "global")
                        txgc = partition_src(tg_c, auto_copy())
                        txrc = partition_dst(tr_C, auto_copy())
                        mask_c = mask(auto_copy(()), [m - bid_x * bm, n - bid_y * bn])
                        copy(auto_copy((bm, bn)), txrc, txgc, mask_c)

        return script_module.ir_module()


def warp_specialized_gemm_split(m, n, k, qdim, kdim, vdim, zdim):
    gemm = WarpSpecializedGemmSplit(m, n, k, qdim, kdim, vdim, zdim)
    return gemm.build()


def data(M, N, K, QKV, Z, trans_a=False, trans_b=False, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    shape_a = (K, M) if trans_a else (M, K)
    shape_b = (N, K) if trans_b else (K, N)
    a = torch.randint(low=lo, high=hi, size=shape_a, dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=shape_b, dtype=dtype, device=device)
    qkv = torch.empty((M, QKV), dtype=dtype, device=device)
    z = torch.empty((M, Z), dtype=dtype, device=device)

    if return_hidet:
        a = hidet.from_torch(a)
        b = hidet.from_torch(b)
        qkv = hidet.from_torch(qkv)
        z = hidet.from_torch(z)

    return a, b, qkv, z


def main(m, n, k, num_k_heads, num_v_heads, head_k_dim, head_v_dim, tp_size, cand=None):
    print(
        f"m: {m}, n: {n}, k: {k}, num_k_heads: {num_k_heads}, num_v_heads: {num_v_heads}, head_k_dim: {head_k_dim}, head_v_dim: {head_v_dim}, tp_size: {tp_size}"
    )
    qdim = head_k_dim
    kdim = head_k_dim
    vdim = head_v_dim * num_v_heads // num_k_heads
    zdim = head_v_dim * num_v_heads // num_k_heads
    qkv = (qdim + kdim + vdim) * num_k_heads // tp_size
    z = zdim * num_k_heads // tp_size
    artifacts = warp_specialized_gemm_split(m, n, k, qdim, kdim, vdim, zdim)
    a, b, qkvt, zt = data(m, n, k, qkv, z, trans_b=True, return_hidet=True)

    best_time = None
    best_i = None
    best_func = None

    for i, (mod, func) in enumerate(artifacts):
        if cand is not None and i != cand:
            continue

        cluster_m = mod._tuning_kwargs["cluster_m"]
        cluster_n = mod._tuning_kwargs["cluster_n"]
        tiled_mma = mod._tuning_kwargs["tiled_mma"]
        k_pipe_max = mod._tuning_kwargs["k_pipe_max"]
        bk = mod._tuning_kwargs["bk"]
        print(f"benchmarking candidate: {i}")
        print(f"cluster_m, cluster_n={cluster_m}, {cluster_n}")
        print(tiled_mma.str_indented())
        print(f"k_pipe_max={k_pipe_max}")
        print(f"bk={bk}")

        def fn():
            func(a, b, qkvt, zt)

        mean, _, _ = bench(fn, ())
    #    mean = do_bench(fn, percentiles=None)
        flops = 2.0 * m * n * k
        memory = f16.nbytes * (m * k + k * n) + f16.nbytes * m * n
        print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
        print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

        if best_time is None:
            best_time = mean
            best_i = i
            best_func = func
        elif mean < best_time:
            best_time = mean
            best_i = i
            best_func = func

    print(f"m: {m}, n: {n}, k: {k}, qkv: {qkv}, z: {z}")
    print(best_i)
    print(best_time)

    func = best_func

    def fn():
        func(a, b, qkvt, zt)

    mean, _, _ = bench(fn, ())
    #mean = do_bench(fn, percentiles=None)
    flops = 2.0 * m * n * k
    memory = f8e4m3.nbytes * (m * k + k * n) + f16.nbytes * m * n
    print("Hexcute: time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    hexcute_mean = mean

    func(a, b, qkvt, zt)
    qkvt = qkvt.torch()
    zt = zt.torch()
    c = torch.concat([qkvt, zt], dim=-1)
    print(c.shape)

    torch_a = a.torch()
    torch_b = b.torch()

    import numpy as np
    from einops import rearrange

    def fn2():
        qkvz = torch_a @ torch_b.T
        new_tensor_shape_qkvz = qkvz.size()[:-1] + (
            num_k_heads // tp_size,
            head_k_dim + head_k_dim + (head_v_dim + head_v_dim) * num_v_heads // num_k_heads,
        )
        qkvz = qkvz.view(*new_tensor_shape_qkvz)
        split_arg_list_qkvz = [head_k_dim, head_k_dim, num_v_heads // num_k_heads * head_v_dim, num_v_heads // num_k_heads * head_v_dim]
        (q, k, v, z) = torch.split(qkvz, split_arg_list_qkvz, dim=2)
        v = v.reshape(v.size(0), -1, head_v_dim)
        z = z.reshape(z.size(0), -1, head_v_dim)
        q, k, v = map(lambda x: rearrange(x, 'l p d -> l (p d)'), (q, k, v))
        qkv = torch.cat((q, k, v), dim=-1)
        return qkv.contiguous(), z.contiguous()

    qkv2, z2 = fn2()
    print(qkv2.size())
    print(z2.size())

    mean, _, _ = bench(fn2, ())
    #mean = do_bench(fn2, percentiles=None)
    cublas_mean = mean
    print(f"cublas:{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")

    options = {"triton.cudagraphs": False, "epilogue_fusion": True, "max_autotune": True}
    fn_inductor = torch.compile(fn2, options=options)
    mean, _, _ = bench(fn_inductor, ()) 
    triton_mean = mean
    # mean
    print(f"inductor: {m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)

    c2 = torch.cat((qkv2, z2.reshape(z2.size(0), -1)), dim=-1)
    np.testing.assert_allclose(actual=c.to(torch.float32).cpu().numpy(), desired=c2.to(torch.float32).cpu().numpy(), rtol=1e-2)
    # c3 = fn3()
    # np.testing.assert_allclose(
    #    actual=c2.to(torch.float32).cpu().numpy(), desired=c3.to(torch.float32).cpu().numpy(), rtol=1e-2
    # )
    return hexcute_mean, cublas_mean, triton_mean


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the performance of the scaled_mm operation")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--search-space", type=int, choices=[1, 2], default=2, help="Search space of Hidet, can be either 1 or 2")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache for generated kernels")
    parser.add_argument("--cand", "-i", type=int, default=None, help="The candidate you want to pick")
    parser.add_argument("--arch", type=int, default=None, help="Compute capability of CUDA")
    parser.add_argument("--debug", "-d", action="store_true", help="whether enabling debug mode or not")
    parser.add_argument("--output", "-o", type=str, default=None, help="output txt")

    args = parser.parse_args()
    if args.cache_dir is not None:
        hidet.option.cache_dir(args.cache_dir)
    hidet.option.search_space(args.search_space)
    if args.debug:
        hidet.option.debug_cache_tuning()
        hidet.option.save_lower_ir(True)

    m, n, k = (args.m, args.n, args.k)
    if args.arch is not None:
        sm_ver = args.arch
    else:
        sm_ver = hidet.option.cuda.get_arch_pair()
        sm_ver = sm_ver[0] * 10 + sm_ver[1]
    print(f"CUDA compute capability: {sm_ver}")

    num_k_heads = 16
    num_v_heads = 32
    tp_size = 4
    head_k_dim = 128
    head_v_dim = 128
    time_hexcute, time_cublas, time_triton = main(m, n, k, num_k_heads, num_v_heads, head_k_dim, head_v_dim, tp_size, cand=args.cand)
