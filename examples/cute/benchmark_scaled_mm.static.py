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


_tiled_mma_lists: List[TiledMma] = []


@initialize()
def register_tiled_mma():
    for warpgroup_m in [1, 2]:
        for n in range(16, 257, 16):
            a = TensorLayout(((128,), (64, 32)), ((0,), (1, 64)))
            b = TensorLayout(((128,), (n, 32)), ((0,), (1, n)))
            c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((128, 1, 16), (64, 8, 512)))
            mma_atom = MmaAtom("warp_group", (64, n, 32), a, b, c, c)
            wg_in_threadblock = Level(
                "warp_group", "thread_block", (warpgroup_m, 1), TensorLayout((warpgroup_m, 1)), (1, 1)
            )
            tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])
            _tiled_mma_lists.append(tiled_mma)


class W8A8ScaledMM:
    def __init__(self, m, n, k, group_n, group_k):
        self.m = m
        self.n = n
        self.k = k
        self.group_n = group_n
        self.group_k = group_k

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


    @tune.space(
        2, cluster_m=[1, 2, 4, 8, 16], cluster_n=[1, 2, 4, 8, 16], tiled_mma=_tiled_mma_lists, k_pipe_max=[4, 5, 6, -1]
    )
    def _candidates(self, cluster_m, cluster_n, tiled_mma: TiledMma, k_pipe_max):
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
        bk = group_k
        tune.check(bn == group_n and bk == group_k)
        tma_copy_tx = (bm * bk + bn * bk) * f8e4m3.nbytes + (bm + (bn // group_k)) * f32.nbytes
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        sm_ver = hidet.option.cuda.get_arch_pair()
        sm_ver = sm_ver[0] * 10 + sm_ver[1]
        smem_limit = smem_limits[sm_ver]
        if k_pipe_max == -1:
            k_pipe_max = int(smem_limit / tma_copy_tx)
        tune.check(tma_copy_tx * k_pipe_max <= smem_limit)
        device_prop = hidet.cuda.properties()
        num_sms = device_prop.multiProcessorCount

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

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: f8e4m3[m, k],
                b_ptr: ~f8e4m3,
                c: bf16[m, n],
                scale_a: f32[m, k // group_k],
                scale_b: f32[n // group_k, k // group_k],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = num_producer_threads + num_consumer_threads  # 8 warps
                attrs.cuda.grid_dim = grid_size, 1, 1
                attrs.cuda.cluster_dim = cluster_size
                attrs.cuda.min_blocks = 1
                attrs.cuda.dynamic_smem_bytes = 0

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

                mbar_tma = make_mbarriers(k_pipe_max)
                mbar_mma = make_mbarriers(k_pipe_max)
                tg_a = tensor_view(a, TensorLayout((m, k), (k, 1)), "global", (bm, k), (bid_x * bm, 0))
                b = as_tensor_pointer(b_ptr, f8e4m3, [n, k])
                tg_b = tensor_view(b, TensorLayout((n, k), (k, 1)), "global", (bn, k), (bid_y * bn, 0))
                tg_sa = tensor_view(
                    scale_a,
                    TensorLayout((m, (bn, k // group_k)), (k // group_k, (0, 1))),
                    "global",
                    (bm, bn * k // group_k),
                    (bid_x * bm, 0),
                )
                tg_sb = tensor_view(
                    scale_b,
                    TensorLayout(((group_k, n // group_k), (bm, k // group_k)), ((0, k // group_k), (0, 1))),
                    "global",
                    (bn, bm * k // group_k),
                    (bid_y * bn, 0),
                )

                ts_sb = make_tensor("float32", TensorLayout((bn, bm, k_pipe_max), (0, 0, 1)), "shared")
                ts_sa = make_tensor("float32", TensorLayout((bm, bn, k_pipe_max), (1, 0, bm)), "shared")
                ts_b = make_tensor(f8e4m3, layout_auto((bn, bk, k_pipe_max)), "shared")
                ts_a = make_tensor(f8e4m3, layout_auto((bm, bk, k_pipe_max)), "shared")

                syncthreads()

                with warp_groups_producer(producer_warpgroups, num_regs=40):
                    smem_pipe_write = 0
                    write_phase = True

                    txga = partition_src(tg_a, auto_copy())
                    txsa = partition_dst(ts_a, auto_copy())
                    txgb = partition_src(tg_b, auto_copy())
                    txsb = partition_dst(ts_b, auto_copy())
                    txgsa = partition_src(tg_sa, auto_copy())
                    txgsb = partition_src(tg_sb, auto_copy())
                    txssa = partition_dst(ts_sa, auto_copy())
                    txssb = partition_dst(ts_sb, auto_copy())

                    k_blocks = cdiv(k, bk)
                    for ko in grid(k_blocks, attrs=unroll):
                        if ko >= k_pipe_max:
                            mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)
                        copy(
                            auto_copy((bm, bk)),
                            txga[:, :, ko],
                            txsa[:, :, smem_pipe_write],
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        copy(
                            auto_copy((bn, bk)),
                            txgb[:, :, ko],
                            txsb[:, :, smem_pipe_write],
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        copy(
                            auto_copy((bm, bn)),
                            txgsa[:, :, ko],
                            txssa[:, :, smem_pipe_write],
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        copy(
                            auto_copy((bn, bm)),
                            txgsb[:, :, ko],
                            txssb[:, :, smem_pipe_write],
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

                with warp_groups_consumer(consumer_warpgroups, num_regs=232):
                    smem_pipe_read = 0
                    read_phase = False
                    smem_pipe_release = 0
                    release_phase = False

                    tr_c_final = make_tensor("float32", layout_auto((bm, bn)), "register")
                    tr_sa = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                    tr_sb = make_tensor("float32", layout_auto((bm, bn), (0, 0)), "register")
                    ts_sbt = transpose(ts_sb, 1, 0, 2)

                    fill(tr_c_final, 0.0)

                    txSa = partition_A(ts_a, tiled_mma)
                    txSb = partition_B(ts_b, tiled_mma)

                    txSsa = partition_src(ts_sa, auto_copy())
                    txSsb = partition_src(ts_sbt, auto_copy())
                    txrsa = partition_dst(tr_sa, auto_copy())
                    txrsb = partition_dst(tr_sb, auto_copy())

                    k_blocks = cdiv(k, bk)
                    k_tiles = cdiv(bk, inst_k)

                    for ko in grid(k_blocks, attrs=unroll):
                        tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
                        mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                        fill(tr_c, 0.0)
                        copy(auto_copy(), txSsa[:, :, smem_pipe_read], txrsa)
                        copy(auto_copy(), txSsb[:, :, smem_pipe_read], txrsb)
                        scale = txrsa * txrsb
                        wgmma_fence_operand(tr_c)
                        wgmma_fence()
                        for ki in range(k_tiles):
                            mma(
                                tiled_mma,
                                tr_c,
                                txSa[:, :, ki, smem_pipe_read],
                                txSb[:, :, ki, smem_pipe_read],
                                tr_c,
                                cluster_layout=cluster_layout,
                            )
                        wgmma_commit_group()
                        wgmma_fence_operand(tr_c)
                        wgmma_wait_group(0)
                        mbarrier_arrive(mbar_mma[smem_pipe_release])
                        tr_c_final = tr_c * scale + tr_c_final

                        smem_pipe_read += 1
                        if smem_pipe_read == k_pipe_max:
                            smem_pipe_read = 0
                            read_phase = not read_phase
                        smem_pipe_release += 1
                        if smem_pipe_release == k_pipe_max:
                            smem_pipe_release = 0
                            release_phase = not release_phase

                    tr_C = rearrange(cast(tr_c_final, bf16), auto_layout, "register")

                    tg_c = tensor_view(
                        c[bid_x * bm : (bid_x + 1) * bm, bid_y * bn : (bid_y + 1) * bn],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )
                    txgc = partition_src(tg_c, auto_copy())
                    txrc = partition_dst(tr_C, auto_copy())
                    mask_c = mask(auto_copy(()), [m - bid_x * bm, n - bid_y * bn])
                    copy(auto_copy((bm, bn)), txrc, txgc, mask_c)

        return script_module.ir_module()


def w8a8_scaled_mm(m, n, k, group_n, group_k):
    scaled_mm_kernel = W8A8ScaledMM(m, n, k, group_n, group_k)
    return scaled_mm_kernel.build()


def f8_quant_data(
    M,
    N,
    K,
    trans_a=False,
    trans_b=False,
    dtype="int8",
    device="cuda",
    return_hidet=False,
    group_m=1,
    group_n=128,
    group_k=64,
):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    shape_a = (K, M) if trans_a else (M, K)
    shape_b = (N, K) if trans_b else (K, N)
    a = torch.randint(low=lo, high=hi, size=shape_a, dtype=torch.float32, device=device).to(dtype=torch.float8_e4m3fn)
    b = torch.randint(low=lo, high=hi, size=shape_b, dtype=torch.float32, device=device).to(dtype=torch.float8_e4m3fn)
    scale_a = torch.randint(low=lo, high=hi, size=(M // group_m, K // group_k), dtype=torch.float32, device=device)
    scale_b = torch.randint(low=lo, high=hi, size=(N // group_n, K // group_k), dtype=torch.float32, device=device)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

    if return_hidet:
        a = hidet.from_torch(a)
        b = hidet.from_torch(b)
        scale_a = hidet.from_torch(scale_a)
        scale_b = hidet.from_torch(scale_b)
        c = hidet.from_torch(c)

    return a, b, scale_a, scale_b, c


def main(m, n, k, group_m, group_n, group_k, cand=None):
    print(f"m: {m}, n: {n}, k: {k}, group_m: {group_m}, group_n: {group_n}, group_k: {group_k}")
    artifacts = w8a8_scaled_mm(m, n, k, group_n, group_k)
    a, b, scale_a, scale_b, c = f8_quant_data(
        m, n, k, trans_b=True, return_hidet=True, group_m=1, group_n=group_n, group_k=group_k
    )

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
        print(f"benchmarking candidate: {i}")
        print(f"cluster_m, cluster_n={cluster_m}, {cluster_n}")
        print(tiled_mma.str_indented())
        print(f"k_pipe_max={k_pipe_max}")

        def fn():
            func(a, b, c, scale_a, scale_b)

        mean = do_bench(fn, percentiles=None)
        flops = 2.0 * m * n * k
        memory = f8e4m3.nbytes * (m * k + k * n) + f16.nbytes * m * n
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

    print(f"m: {m}, n: {n}, k: {k}, group_m: {group_m}, group_n: {group_n}, group_k: {group_k}")
    print(best_i)
    print(best_time)

    func = best_func

    def fn():
        func(a, b, c, scale_a, scale_b)

    mean = do_bench(fn, percentiles=None)
    flops = 2.0 * m * n * k
    memory = f8e4m3.nbytes * (m * k + k * n) + f16.nbytes * m * n
    print("Hexcute: time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    hexcute_mean = mean

    func(a, b, c, scale_a, scale_b)

    torch_a = a.torch()
    torch_b = b.torch()
    torch_scale_a = scale_a.torch().transpose(0, 1).contiguous()
    torch_scale_b = scale_b.torch()

    try:
        from vllm import _custom_ops as ops

        cutlass_scaled_fp8_gemm = ops.cutlass_scaled_mm
        cutlass_scaled_fp8_gemm(torch_a, torch_b.T, torch_scale_a.T, torch_scale_b.T, out_dtype=torch.bfloat16)
    except (ImportError, ValueError):
        cutlass_scaled_fp8_gemm = None

    from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_block_fp8_matmul

    vllm_scaled_fp8_gemm = w8a8_block_fp8_matmul
    vllm_scaled_fp8_gemm(
        torch_a, torch_b, torch_scale_a.T, torch_scale_b, block_size=[group_k, group_k], output_dtype=torch.bfloat16
    )
    #vllm_scaled_fp8_gemm = None

    def fn2():
        return cutlass_scaled_fp8_gemm(torch_a, torch_b.T, torch_scale_a.T, torch_scale_b.T, out_dtype=torch.bfloat16)

    def fn3():
        return vllm_scaled_fp8_gemm(
            torch_a, torch_b, torch_scale_a.T, torch_scale_b, block_size=[group_k, group_k], output_dtype=torch.bfloat16
        )

    if cutlass_scaled_fp8_gemm is not None:
        mean = do_bench(fn2, percentiles=None)
        cutlass_mean = mean
        print(f"cutlass:{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")

    if vllm_scaled_fp8_gemm is not None:
        mean = do_bench(fn3, percentiles=None)
        triton_mean = mean
        print(f"triton: {m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)

    if cutlass_scaled_fp8_gemm is not None:
        c2 = fn2()
        np.testing.assert_allclose(
            actual=c.torch().to(torch.float32).cpu().numpy(), desired=c2.to(torch.float32).cpu().numpy(), rtol=1e-2
        )
        if vllm_scaled_fp8_gemm is not None:
            c3 = fn3()
            np.testing.assert_allclose(
                actual=c2.to(torch.float32).cpu().numpy(), desired=c3.to(torch.float32).cpu().numpy(), rtol=1e-2
            )
    return hexcute_mean, cutlass_mean, triton_mean


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the performance of the scaled_mm operation")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--group-m", type=int, default=1)
    parser.add_argument("--group-n", type=int, default=128)
    parser.add_argument("--group-k", type=int, default=128)
    parser.add_argument(
        "--search-space", type=int, choices=[1, 2], default=2, help="Search space of Hidet, can be either 1 or 2"
    )
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
    hidet.option.num_local_workers(1)

    m, n, k, group_m, group_n, group_k = (args.m, args.n, args.k, args.group_m, args.group_n, args.group_k)
    if args.arch is not None:
        sm_ver = args.arch
    else:
        sm_ver = hidet.option.cuda.get_arch_pair()
        sm_ver = sm_ver[0] * 10 + sm_ver[1]
    print(f"CUDA compute capability: {sm_ver}")

    weight_shapes = [[2048, 7168], [24576, 1536], [32768, 512], [4096, 7168], [7168, 2048], [4096, 16384], [4096, 4096]]
    from tabulate import tabulate

    records = []
    headers = ["mxnxk", "triton", "cutlass", "hexcute", "flops_triton", "flops_cutlass", "flops_hexcute"]

    triton = []
    cutlass = []
    hexcute = []

    for m in [32, 64, 128, 2048, 4096]:
        for n, k in weight_shapes:
            flops = 2.0 * m * n * k
            time_hexcute, time_cutlass, time_triton = main(m, n, k, group_m, group_n, group_k)
            flops_hexcute = flops / time_hexcute / 1e9
            flops_cutlass = flops / time_cutlass / 1e9
            flops_triton = flops / time_triton / 1e9
            shape = f"{m}x{n}x{k}"
            records.append([shape, time_triton, time_cutlass, time_hexcute, flops_triton, flops_cutlass, flops_hexcute])
            triton.append(flops_triton)
            cutlass.append(flops_cutlass)
            hexcute.append(flops_hexcute)
    
    with open(args.output, "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc     
    methods = ['Hexcute', 'FlashAttention', 'FlashInfer', 'Triton', 'CUTLASS', 'cuBLAS']
    
    clist = ['#b5739d', '#7ea6e0', '#67ab9f', '#ea6b66', '#ffb570', '#97d077']
    my_colors = {}
    for i, method in enumerate(methods):
        my_colors[method] = clist[i]
 
    rc('font', **{'family': 'sans-serif', 'size': 25})

    # Data for each method
    methods = ['Triton', 'CUTLASS', 'Hexcute']

    # Latency for each matrix type
    triton = [  9.78,  24.40,  11.93,  19.57,  10.10,  17.32,  10.22, 19.17,   48.80,  23.86, 38.74, 19.99,  35.35, 20.64, 38.74,  92.92, 44.28,  73.69,  40.41, 73.10, 40.90,   150.70, 214.45,   188.79,  185.30,  213.22,  174.19,  195.78,  178.43,  221.04,  193.03,  201.10, 218.256, 196.27, 217.81]
    cutlass = [14.91,  43.92,  20.65,  29.82,  18.07,  50.53,  19.88, 29.83,   89.48,  41.29, 60.61, 36.14, 101.06, 36.40, 60.61, 178.96, 82.60, 119.30, 72.27, 202.12, 79.536, 522.865, 687.194, 467.479, 643.096, 683.290, 643.742, 624.722, 639.675, 693.357, 475.567, 675.612, 699.180, 616.318, 690.648]
    hexcute = [18.42, 56.184, 26.189, 38.348, 24.090, 68.174, 24.970, 38.34, 102.805, 52.377, 75.16, 46.98, 122.71, 47.72, 79.96, 197.22, 89.48, 156.59, 96.36, 241.97, 95.44,  583.781, 566.37,  234.53,  683.29,  583.781, 656.581, 613.566, 653.581, 578.014, 240.277, 699.180, 598.303, 695.893, 657.602]

    fig, ax = plt.subplots(1, 1, figsize=(30, 4))
    
    print(len(triton))
    print(len(cutlass))
    print(len(hexcute))
    categories = [f"M{m}" for m in range(len(cutlass))]
    width = 0.2         # Width of the bars
    N = len(categories)
    ind = np.arange(N)  # X locations for the groups
    
    import numpy as np
    cmap = plt.get_cmap('gnuplot')
    ll = cmap.N*8//9
    len_methods = len(methods)
    indices = np.linspace(ll//5, ll, len_methods)
    
    gap = 0.012
    i = 0
    ax.bar(ind + (i + 0.5) * (width + gap), triton, width, label=methods[i], color=my_colors[methods[i]])
    i = 1
    ax.bar(ind + (i + 0.5) * (width + gap), cutlass, width, label=methods[i], color=my_colors[methods[i]])
    i = 2
    ax.bar(ind + (i + 0.5) * (width + gap), hexcute, width, label=methods[i], color=my_colors[methods[i]])
 
    ax.set_ylabel('Throughput (TFLOPS)', fontsize=18)
    ax.set_ylim(0, 750)
    ax.set_xlabel('FP8 Block Scaled GEMM Layers', fontsize=18)
    ax.set_xticks(ind + (len(methods) * width) / 2)
    ax.set_yticks(np.arange(0, 750, 75))
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    ax.set_xticklabels(categories, fontsize=18)
    # title_loc = -0.2
    #ax.set_title('FP16xINT4 MoE Layer', fontsize=18)
    ax.yaxis.grid(True, linestyle='dotted')

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    x = set()
    lins = []
    labs = []
    for li, la in zip(lines, labels):
        if la in x:
            continue
        x.add(la)
        lins.append(li)
        labs.append(la)
    fig.legend(lins, labs, loc='upper left', bbox_to_anchor=(0.053, 0.95), fontsize=14, ncols=3)

    fig.subplots_adjust(
            top=0.94,
            bottom=0.173,
            left=0.053,
            right=0.99,
            hspace=0.2,
            wspace=0.2
        )
    # Adjust layout to prevent clipping of tick-labels
    plt.savefig(args.output.replace(".txt", ".pdf"), dpi=300, bbox_inches='tight')


def func(
    a: f8e4m3[m, k],
    b_ptr: ~f8e4m3,
    c: bf16[m, n],
    scale_a: f32[m, k // group_k],
    scale_b: f32[n // group_k, k // group_k],
):
    attrs.func_kind = "cuda_kernel"
    attrs.cuda.block_dim = num_producer_threads + num_consumer_threads  # 8 warps
    attrs.cuda.grid_dim = grid_size, 1, 1
    attrs.cuda.cluster_dim = cluster_size
    attrs.cuda.min_blocks = 1
    attrs.cuda.dynamic_smem_bytes = 0

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

    mbar_tma = make_mbarriers(k_pipe_max)
    mbar_mma = make_mbarriers(k_pipe_max)
    tg_a = tensor_view(a, TensorLayout((m, k), (k, 1)), "global", (bm, k), (bid_x * bm, 0))
    b = as_tensor_pointer(b_ptr, f8e4m3, [n, k])
    tg_b = tensor_view(b, TensorLayout((n, k), (k, 1)), "global", (bn, k), (bid_y * bn, 0))
    tg_sa = tensor_view(
        scale_a,
        TensorLayout((m, (bn, k // group_k)), (k // group_k, (0, 1))),
        "global",
        (bm, bn * k // group_k),
        (bid_x * bm, 0),
    )
    tg_sb = tensor_view(
        scale_b,
        TensorLayout(((group_k, n // group_k), (bm, k // group_k)), ((0, k // group_k), (0, 1))),
        "global",
        (bn, bm * k // group_k),
        (bid_y * bn, 0),
    )

    ts_sb = make_tensor("float32", TensorLayout((bn, bm, k_pipe_max), (0, 0, 1)), "shared")
    ts_sa = make_tensor("float32", TensorLayout((bm, bn, k_pipe_max), (1, 0, bm)), "shared")
    ts_b = make_tensor(f8e4m3, layout_auto((bn, bk, k_pipe_max)), "shared")
    ts_a = make_tensor(f8e4m3, layout_auto((bm, bk, k_pipe_max)), "shared")

    syncthreads()

    with warp_groups_producer(producer_warpgroups, num_regs=40):
        smem_pipe_write = 0
        write_phase = True

        txga = partition_src(tg_a, auto_copy())
        txsa = partition_dst(ts_a, auto_copy())
        txgb = partition_src(tg_b, auto_copy())
        txsb = partition_dst(ts_b, auto_copy())
        txgsa = partition_src(tg_sa, auto_copy())
        txgsb = partition_src(tg_sb, auto_copy())
        txssa = partition_dst(ts_sa, auto_copy())
        txssb = partition_dst(ts_sb, auto_copy())

        k_blocks = cdiv(k, bk)
        for ko in grid(k_blocks, attrs=unroll):
            if ko >= k_pipe_max:
                mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)
            copy(auto_copy((bm, bk)), txga[:, :, ko], txsa[:, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])
            copy(auto_copy((bn, bk)), txgb[:, :, ko], txsb[:, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])
            copy(auto_copy((bm, bn)), txgsa[:, :, ko], txssa[:, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])
            copy(auto_copy((bn, bm)), txgsb[:, :, ko], txssb[:, :, smem_pipe_write], mbarrier=mbar_tma[smem_pipe_write])

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

    with warp_groups_consumer(consumer_warpgroups, num_regs=232):
        smem_pipe_read = 0
        read_phase = False
        smem_pipe_release = 0
        release_phase = False

        tr_c_final = make_tensor("float32", layout_auto((bm, bn)), "register")
        tr_sa = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
        tr_sb = make_tensor("float32", layout_auto((bm, bn), (0, 0)), "register")
        ts_sbt = transpose(ts_sb, 1, 0, 2)

        fill(tr_c_final, 0.0)

        txSa = partition_A(ts_a, tiled_mma)
        txSb = partition_B(ts_b, tiled_mma)

        txSsa = partition_src(ts_sa, auto_copy())
        txSsb = partition_src(ts_sbt, auto_copy())
        txrsa = partition_dst(tr_sa, auto_copy())
        txrsb = partition_dst(tr_sb, auto_copy())

        k_blocks = cdiv(k, bk)
        k_tiles = cdiv(bk, inst_k)

        for ko in grid(k_blocks, attrs=unroll):
            tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
            mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
            fill(tr_c, 0.0)
            copy(auto_copy(), txSsa[:, :, smem_pipe_read], txrsa)
            copy(auto_copy(), txSsb[:, :, smem_pipe_read], txrsb)
            scale = txrsa * txrsb
            wgmma_fence_operand(tr_c)
            wgmma_fence()
            for ki in range(k_tiles):
                mma(
                    tiled_mma,
                    tr_c,
                    txSa[:, :, ki, smem_pipe_read],
                    txSb[:, :, ki, smem_pipe_read],
                    tr_c,
                    cluster_layout=cluster_layout,
                )
            wgmma_commit_group()
            wgmma_fence_operand(tr_c)
            wgmma_wait_group(0)
            mbarrier_arrive(mbar_mma[smem_pipe_release])
            tr_c_final = tr_c * scale + tr_c_final

            smem_pipe_read += 1
            if smem_pipe_read == k_pipe_max:
                smem_pipe_read = 0
                read_phase = not read_phase
            smem_pipe_release += 1
            if smem_pipe_release == k_pipe_max:
                smem_pipe_release = 0
                release_phase = not release_phase

        tr_C = rearrange(cast(tr_c_final, bf16), auto_layout, "register")

        tg_c = tensor_view(
            c[bid_x * bm : (bid_x + 1) * bm, bid_y * bn : (bid_y + 1) * bn], TensorLayout((bm, bn), (n, 1)), "global"
        )
        txgc = partition_src(tg_c, auto_copy())
        txrc = partition_dst(tr_C, auto_copy())
        mask_c = mask(auto_copy(()), [m - bid_x * bm, n - bid_y * bn])
        copy(auto_copy((bm, bn)), txrc, txgc, mask_c)
