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
from hidet.ir.primitives.cuda.copy_tma import (
    copy_bulk_commit_group,
    copy_bulk_wait_group,
)
from hidet.ir.primitives.cuda.barrier import fence_view_async_shared


epilogue_subtiling = True


_tiled_mma_lists: List[TiledMma] = []


@initialize()
def register_tiled_mma():
    non_power_of_two = True
    if non_power_of_two:
        ns = [32, 64, 96] + list(range(128, 257, 16))
    else:
        ns = [32, 64, 128, 256]
    for warpgroup_m in [1, 2]:
        for n in ns:
            a = TensorLayout(((128,), (64, 16)), ((0,), (1, 64)))
            b = TensorLayout(((128,), (n, 16)), ((0,), (1, n)))
            c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((128, 1, 16), (64, 8, 512)))
            mma_atom = MmaAtom("warp_group", (64, n, 16), a, b, c, c)
            wg_in_threadblock = Level(
                "warp_group", "thread_block", (warpgroup_m, 1), TensorLayout((warpgroup_m, 1)), (1, 1)
            )
            tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])
            _tiled_mma_lists.append(tiled_mma)


class WarpSpecializedGemm:
    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k

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

        for _ in tqdm(
            parallel_imap_2ndlevel(build_job, jobs, is_remote_allowed=True), desc="Compiling", total=len(jobs), ncols=80
        ):
            pass
        funcs = [load_compiled_module(output_dir) for output_dir in output_dirs]
        return list(zip(ir_modules, funcs))

    @tune.space(
        2, cluster_m=[1, 2, 4, 8, 16], cluster_n=[1, 2, 4, 8, 16], tiled_mma=_tiled_mma_lists, k_pipe_max=[4, 5, 6, -1], bk=[64]
    )
    def _candidates(self, cluster_m, cluster_n, tiled_mma: TiledMma, k_pipe_max, bk):
        m, n, k = self.m, self.n, self.k
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
        tune.check((m < 128 and bm == 64) or (m >= 128 and ((m % 128 == 0 and bm == 128) or (m % 128 != 0))))

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
        tune.check(cluster_m == 1 or cluster_n == 1)
        unroll = f"u{k_pipe_max}"
 
        with hidet.script_module() as script_module:
            
            @hidet.script
            def func(a: f16[m, k], b_ptr: ~f16, c: f16[m, n]):
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
                            mma(
                                tiled_mma,
                                tr_c,
                                txSa[:, :, ki, smem_pipe_read],
                                txSb[:, :, ki, smem_pipe_read],
                                tr_c,
                                cluster_layout=cluster_layout,
                            )
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
                    if epilogue_subtiling:
                        # Convert result to FP16 and prepare for global memory write
                        ts_c = make_tensor("float16", layout_auto((bm, 16)), "shared")
                        tcrc = partition_src(cast(tr_c, f16), auto_copy())
                        tcsc = partition_dst(ts_c, auto_copy())
                        tg_c = tensor_view(c, TensorLayout((m, n), (n, 1)), "global", (bm, bn), (bid_x * bm, bid_y * bn))
                        tCsC = partition_src(ts_c, auto_copy())
                        tCgC = partition_dst(tg_c, auto_copy())
                        epilogue_stages = bn // 16
                        for i in range(epilogue_stages):
                            copy(auto_copy((bm, 16)), tcrc[:, :, i], tcsc)
                            syncthreads()
                            fence_view_async_shared()
                            copy(auto_copy((bm, 16)), tCsC, tCgC[:, :, i])
                            copy_bulk_commit_group()
                            copy_bulk_wait_group(0)
                            syncthreads()
                            fence_view_async_shared()
                    else:
                        # Convert result to FP16 and prepare for global memory write
                        tr_C = rearrange(cast(tr_c, f16), auto_layout, "register")

                        # Write result back to global memory
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


def warp_specialized_gemm(m, n, k):
    gemm = WarpSpecializedGemm(m, n, k)
    return gemm.build()


def data(M, N, K, trans_a=False, trans_b=False, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    shape_a = (K, M) if trans_a else (M, K)
    shape_b = (N, K) if trans_b else (K, N)
    a = torch.randint(low=lo, high=hi, size=shape_a, dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=shape_b, dtype=dtype, device=device)
    c = torch.empty((M, N), dtype=dtype, device=device)

    if return_hidet:
        a = hidet.from_torch(a)
        b = hidet.from_torch(b)
        c = hidet.from_torch(c)

    return a, b, c


non_power_of_two = 0


def main(m, n, k, cand=None):
    print(f"m: {m}, n: {n}, k: {k}")
    import time
    start_time = time.time()
    artifacts = warp_specialized_gemm(m, n, k)
    print("--- Hexcute: compilation time: %s seconds ---" % (time.time() - start_time))
    a, b, c = data(m, n, k, trans_b=True, return_hidet=True)

    best_time = None
    best_i = None
    best_func = None
    best_bn = None

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

        c_tile_shape, _ = tiled_mma.c_tv_layout()
        _, bn = c_tile_shape

        def fn():
            func(a, b, c)

        mean = do_bench(fn, percentiles=None)
        flops = 2.0 * m * n * k
        memory = f16.nbytes * (m * k + k * n) + f16.nbytes * m * n
        print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
        print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

        if best_time is None:
            best_time = mean
            best_i = i
            best_func = func
            best_bn = bn
        elif mean < best_time:
            best_time = mean
            best_i = i
            best_func = func
            best_bn = bn

    print(f"m: {m}, n: {n}, k: {k}")
    print(best_i)
    print(best_time)
    if best_bn & (best_bn - 1) != 0:
        global non_power_of_two
        non_power_of_two += 1 

    func = best_func

    def fn():
        func(a, b, c)

    mean = do_bench(fn, percentiles=None)
    flops = 2.0 * m * n * k
    memory = f8e4m3.nbytes * (m * k + k * n) + f16.nbytes * m * n
    print("Hexcute: time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    hexcute_mean = mean

    func(a, b, c)

    torch_a = a.torch()
    torch_b = b.torch()

    def fn2():
        return torch_a @ torch_b.T

    from matmul import matmul as triton_matmul

    def fn3():
        return triton_matmul(torch_a, torch_b.T)
    
    mean = do_bench(fn2, percentiles=None)
    cublas_mean = mean
    print(f"cublas:{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")

    mean = do_bench(fn3, percentiles=None)
    triton_mean = mean
    print(f"triton: {m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)

    c2 = fn2()
    np.testing.assert_allclose(
        actual=c.torch().to(torch.float32).cpu().numpy(), desired=c2.to(torch.float32).cpu().numpy(), rtol=1e-2
    )
    c3 = fn3()
    np.testing.assert_allclose(
        actual=c2.to(torch.float32).cpu().numpy(), desired=c3.to(torch.float32).cpu().numpy(), rtol=1e-2
    )
    return hexcute_mean, cublas_mean, triton_mean


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the performance of the scaled_mm operation")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
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

    m, n, k = (args.m, args.n, args.k)
    if args.arch is not None:
        sm_ver = args.arch
    else:
        sm_ver = hidet.option.cuda.get_arch_pair()
        sm_ver = sm_ver[0] * 10 + sm_ver[1]
    print(f"CUDA compute capability: {sm_ver}")

    weight_shapes = [
        [6144, 4096],
        [4096, 14336],
        [14336, 4096],
        [4096, 4096],
        [10240, 8192],
        [8192, 28672],
        [8192, 8192],
        [28672, 8192],
    ]
    from tabulate import tabulate

    records = []
    headers = ["mxnxk", "triton", "cublas", "hexcute", "flops_triton", "flops_cublas", "flops_hexcute"]
    records = []

    triton = []
    cublas = []
    hexcute = []
       
    print(non_power_of_two) 
    for m in [32, 64, 128, 2048, 4096]:
        for n, k in weight_shapes:
            time_hexcute, time_cublas, time_triton = main(m, n, k, cand=args.cand)
            shape = f"{m}x{n}x{k}"
            flops = 2.0 * m * n * k
            flops_hexcute = flops / time_hexcute / 1e9
            flops_cublas = flops / time_cublas / 1e9
            flops_triton = flops / time_triton / 1e9
            records.append([shape, time_triton, time_cublas, time_hexcute, flops_triton, flops_cublas, flops_hexcute])
            triton.append(flops_triton)
            cublas.append(flops_cublas)
            hexcute.append(flops_hexcute)

    with open(args.output, "w") as f:
       f.write(
           tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
       )

    methods = ['Hexcute', 'FlashAttention', 'FlashInfer', 'Triton', 'CUTLASS', 'cuBLAS']
    clist = ['#b5739d', '#7ea6e0', '#67ab9f', '#ea6b66', '#ffb570', '#97d077']
    my_colors = {}
    for i, method in enumerate(methods):
        my_colors[method] = clist[i]
   
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc     
    rc('font', **{'family': 'sans-serif', 'size': 25})
    import numpy as np

    # Data for each method
    methods = ['Triton', 'cuBLAS', 'Hexcute']

    fig, ax = plt.subplots(1, 1, figsize=(30, 4))
    
    print(len(triton))
    print(len(cublas))
    print(len(hexcute))
    categories = [f"M{m}" for m in range(len(cublas))]
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
    ax.bar(ind + (i + 0.5) * (width + gap), cublas, width, label=methods[i], color=my_colors[methods[i]])
    i = 2
    ax.bar(ind + (i + 0.5) * (width + gap), hexcute, width, label=methods[i], color=my_colors[methods[i]])

    max_y = max(max(triton), max(cublas), max(hexcute)) + 55
    ax.set_ylabel('Throughput (TFLOPS)', fontsize=18)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('F16 Warp Specialized GEMM Layers', fontsize=18)
    ax.set_xticks(ind + (len(methods) * width) / 2)
    ax.set_yticks(np.arange(0, max_y, 55))
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
    plt.savefig(args.output.replace('.txt', '.pdf'), dpi=300, bbox_inches='tight')