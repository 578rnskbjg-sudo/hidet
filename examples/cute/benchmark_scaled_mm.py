from typing import List
from hidet.ir.cute.contexts import warp_groups_consumer, warp_groups_producer
import torch
import pytest

from hidet.lang.types import f8e4m3, f8e5m2, bf16, f16, f32, i32
from hidet.lang import attrs, grid
from hidet.lang.cuda import blockIdx, threadIdx
from hidet.ir.expr import symbol_var
from hidet.ir.primitives.cuda.wgmma import wgmma_fence, wgmma_commit_group, wgmma_wait_group
from hidet.lang.cuda import syncthreads, cp_async_commit_group, cp_async_wait_group
from hidet.ir.primitives.cuda.copy_tma import copy_bulk_commit_group, copy_bulk_wait_group
from hidet.lang.constructs.declare import as_tensor_pointer
from hidet.utils.py import cdiv
from hidet.ir.primitives.cuda.atomic import atomic_add
from hidet.ir.primitives.cuda.mutex import acquire_seq_semaphore, release_seq_semaphore

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
        for n in [8] + list(range(16, 257, 16)):
            a = TensorLayout(((128,), (n, 32)), ((0,), (1, n)))
            b = TensorLayout(((128,), (64, 32)), ((0,), (1, 64)))
            c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((2, n, 16 * n), (1, 8 * n, 8)))
            mma_atom = MmaAtom("warp_group", (n, 64, 32), a, b, c, c)
            wg_in_threadblock = Level("warp_group", "thread_block", (1, warpgroup_m), TensorLayout((1, warpgroup_m)), (1, 1))
            tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])
            _tiled_mma_lists.append(tiled_mma)
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
    M_BINS = [8, 16, 32, 128]

    def __init__(self, n, k, group_n, group_k):
        self.n = n
        self.k = k
        self.group_n = group_n
        self.group_k = group_k
        self.cache = {}
        self._compile()

    def __call__(
        self, q_input: torch.Tensor, weight: torch.Tensor, x_scale: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        M = q_input.shape[0]
        m_clip = min(max(M, 8), 4096)
        m_roundup = min(i for i in self.M_BINS if i >= m_clip)
        assert m_roundup in self.M_BINS
        bm, bn, parallel_k_parts, func = self.cache[m_roundup]
        from hidet.ffi import runtime_api

        N = self.n
        runtime_api.set_symbol_value("m", M)
        device = "cuda"
        c = torch.empty((M, N), dtype=torch.bfloat16, device=device)
        if parallel_k_parts > 1:
            N = weight.shape[0]
            assert weight.shape[1] == self.k and N == self.n
            grid_m = cdiv(M, bm)
            grid_n = cdiv(N, bn)
            c_partial = torch.empty((parallel_k_parts, M, N), dtype=torch.float32, device=device)
            lock = torch.zeros((grid_m, grid_n), dtype=torch.int32, device=device)
            func(q_input, weight, c, x_scale, weight_scale, c_partial, lock)
        else:
            func(q_input, weight, c, x_scale, weight_scale)
        return c

    def modules(self):
        return tune.extract_ir_modules(self.scaled_mm_split_k)
        return tune.extract_ir_modules(self.scaled_mm) + tune.extract_ir_modules(self.scaled_mm_split_k)

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

    def _create_fake_data(self):
        N = self.n
        K = self.k
        group_n = self.group_n
        group_k = self.group_k
        device = "cuda"
        lo = -3
        hi = 3
        shape_b = (N, K)
        b = torch.randint(low=lo, high=hi, size=shape_b, dtype=torch.float32, device=device).to(
            dtype=torch.float8_e4m3fn
        )
        scale_b = torch.randint(low=lo, high=hi, size=(N // group_n, K // group_k), dtype=torch.float32, device=device)

        return b, scale_b

    def _compile(self):
        artifacts = self.build()

        best_time = None
        best_i = None
        best_parallel_k_parts = None
        best_bm = None
        best_bn = None
        best_func = None

        for M in self.M_BINS:
            K = self.k
            N = self.n
            group_n = self.group_n
            group_k = self.group_k
            b, scale_b = self._create_fake_data()
            for i, (mod, func) in enumerate(artifacts):
                cluster_m = mod._tuning_kwargs["cluster_m"]
                cluster_n = mod._tuning_kwargs["cluster_n"]
                tiled_mma = mod._tuning_kwargs["tiled_mma"]
                k_pipe_max = mod._tuning_kwargs["k_pipe_max"]
                c_shape, c_tv = tiled_mma.c_tv_layout()
                bm, bn = c_shape
                if bm > M or bm < M / 2:
                    continue
                if "parallel_k_parts" in mod._tuning_kwargs:
                    parallel_k_parts = mod._tuning_kwargs["parallel_k_parts"]
                else:
                    parallel_k_parts = 1
                grid_m = cdiv(M, bm)
                grid_n = cdiv(N, bn)
                print(f"benchmarking candidate: {i}")
                print(f"cluster_m, cluster_n={cluster_m}, {cluster_n}")
                print(tiled_mma.str_indented())
                print(f"k_pipe_max={k_pipe_max}")

                device = "cuda"
                lo = -3
                hi = 3
                a = torch.randint(low=lo, high=hi, size=(M, K), dtype=torch.float32, device=device).to(
                    dtype=torch.float8_e4m3fn
                )
                scale_a = torch.randint(
                    low=lo, high=hi, size=(M // group_m, K // group_k), dtype=torch.float32, device=device
                )
                c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
                lock = torch.zeros((grid_m, grid_n), dtype=torch.int32, device=device)
                c_partial = torch.empty((parallel_k_parts, M, N), dtype=torch.float32, device=device)
                from hidet.ffi import runtime_api

                runtime_api.set_symbol_value("m", M)

                def fn():
                    if parallel_k_parts > 1:
                        lock.zero_()
                        func(a, b, c, scale_a, scale_b, c_partial, lock)
                    else:
                        func(a, b, c, scale_a, scale_b)

                mean = do_bench(fn, percentiles=None)
                flops = 2.0 * M * N * K
                memory = f8e4m3.nbytes * (M * K + K * N) + f16.nbytes * M * N
                print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
                print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

                if best_time is None:
                    best_time = mean
                    best_i = i
                    best_func = func
                    best_bm = bm
                    best_bn = bn
                    best_parallel_k_parts = parallel_k_parts
                elif mean < best_time:
                    best_time = mean
                    best_i = i
                    best_func = func
                    best_bm = bm
                    best_bn = bn
                    best_parallel_k_parts = parallel_k_parts

            print(f"m: {M}, n: {self.n}, k: {self.k}, group_n: {self.group_n}, group_k: {self.group_k}")
            print(best_i)
            print(best_time)

            self.cache[M] = (best_bm, best_bn, best_parallel_k_parts, best_func)

    @tune.space(
        2, cluster_m=[1, 2, 4, 8, 16], cluster_n=[1, 2, 4, 8, 16], tiled_mma=_tiled_mma_lists, k_pipe_max=[4, 5, 6, -1]
    )
    def scaled_mm(self, cluster_m, cluster_n, tiled_mma: TiledMma, k_pipe_max):
        m = symbol_var("m")
        n = self.n
        k = self.k
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
        grid_m = cdiv(m, bm)
        grid_n = cdiv(n, bn)
        grid_size = grid_m * grid_n
        #cluster_layout = TensorLayout((cluster_m, cluster_n), (1, cluster_m))
        #cluster_id2mn = right_inverse(cluster_layout)
        #cluster_size = cluster_layout.size()
        #tune.check(cluster_size <= 16)
        #cluster_shape = product_each(cluster_layout.shape_tuple)
        #cluster_m, cluster_n = cluster_shape
        #grid_size = cluster_size * cdiv(m, cluster_m * bm) * cdiv(n, cluster_n * bn)
        #cta_ms = cdiv(m, cluster_m * bm) * cluster_m
        #cta_ns = cdiv(n, cluster_n * bn) * cluster_n
        #min_cta_dim = min(cta_ms, cta_ns)
        #if min_cta_dim >= 6:
        #    log_swizzle_size = 3
        #elif min_cta_dim >= 3:
        #    log_swizzle_size = 2
        #elif min_cta_dim >= 2:
        #    log_swizzle_size = 1
        #else:
        #    log_swizzle_size = 0
        #cluster_blk_major = cdiv(n, cluster_n * bn)
        #tune.check(num_sms % cluster_size == 0)
        unroll = f"u{k_pipe_max}"
        tune.check(cluster_m == 1 and cluster_n == 1)
        cluster_size = cluster_m * cluster_n

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
                bid_x = pid // grid_n
                bid_y = pid % grid_n
                ## Block indices for grid-level parallelism
                #cluster_id = pid % cluster_size
                #cluster_mn = cluster_id2mn(cluster_id)
                #cluster_index_m = cluster_mn // cluster_n
                #cluster_index_n = cluster_mn % cluster_n
                #grid_id = pid // cluster_size
                #offset = grid_id & ((1 << log_swizzle_size) - 1)
                #extra = grid_id >> log_swizzle_size
                #cluster_idx_minor_div_swizzle = extra // cluster_blk_major
                #cluster_idx_major = extra % cluster_blk_major
                #cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset
                #bid_x = cluster_idx_minor * cluster_m + cluster_index_m
                #bid_y = cluster_idx_major * cluster_n + cluster_index_n

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

    @tune.space(
        2,
        cluster_m=[1, 2, 4, 8, 16],
        cluster_n=[1, 2, 4, 8, 16],
        tiled_mma=_tiled_mma_lists,
        k_pipe_max=[4, 5, 6, -1],
        parallel_k_parts=[2, 4],
    )
    def scaled_mm_split_k(self, cluster_m, cluster_n, tiled_mma: TiledMma, k_pipe_max, parallel_k_parts):
        m = symbol_var("m")
        n = self.n
        k = self.k
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
        tune.check(k % parallel_k_parts == 0 and (k // parallel_k_parts) % bk == 0)
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
        tune.check(cluster_m == 1 and cluster_n == 1)

        num_consumer_threads = threads
        num_producer_threads = 128

        if num_consumer_threads == 256:
            producer_warpgroups = [2]
            consumer_warpgroups = [0, 1]
        elif num_consumer_threads == 128:
            producer_warpgroups = [1]
            consumer_warpgroups = [0]

        # This is a tunable knob for cluster layout
        grid_m = cdiv(m, bm)
        grid_n = cdiv(n, bn)
        grid_size = grid_m * grid_n
        unroll = f"u{k_pipe_max}"
        BK_per_tile = k // parallel_k_parts
        cluster_size = cluster_m * cluster_n

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: f8e4m3[m, k],
                b_ptr: ~f8e4m3,
                c: bf16[m, n],
                scale_a: f32[m, k // group_k],
                scale_b: f32[n // group_k, k // group_k],
                c_partial: f32[parallel_k_parts, m, n],
                lock: i32[grid_m, grid_n],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = num_producer_threads + num_consumer_threads  # 8 warps
                attrs.cuda.grid_dim = grid_size * parallel_k_parts, 1
                attrs.cuda.cluster_dim = cluster_size
                attrs.cuda.min_blocks = 1
                attrs.cuda.dynamic_smem_bytes = 0

                pid = blockIdx.x
                # Block indices for grid-level parallelism
                k_part = pid % parallel_k_parts
                k_start = k_part * BK_per_tile
                pid_mn = pid // parallel_k_parts
                bid_x = pid_mn // grid_n
                bid_y = pid_mn % grid_n

                mbar_tma = make_mbarriers(k_pipe_max)
                mbar_mma = make_mbarriers(k_pipe_max)
                tg_a = tensor_view(a, TensorLayout((m, k), (k, 1)), "global", (bm, k), (bid_x * bm, k_start))
                b = as_tensor_pointer(b_ptr, f8e4m3, [n, k])
                tg_b = tensor_view(b, TensorLayout((n, k), (k, 1)), "global", (bn, k), (bid_y * bn, k_start))
                tg_sa = tensor_view(
                    scale_a,
                    TensorLayout((m, (bn, k // group_k)), (k // group_k, (0, 1))),
                    "global",
                    (bm, bn * k // group_k),
                    (bid_x * bm, k_start // group_k * bn),
                )
                tg_sb = tensor_view(
                    scale_b,
                    TensorLayout(((group_k, n // group_k), (bm, k // group_k)), ((0, k // group_k), (0, 1))),
                    "global",
                    (bn, bm * k // group_k),
                    (bid_y * bn, k_start // group_k * bm),
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

                    k_blocks = cdiv(BK_per_tile, bk)
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

                    k_blocks = cdiv(BK_per_tile, bk)
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

                    tr_C = rearrange(tr_c_final, auto_layout, "register")
                    lc = ~lock[bid_x, bid_y]
                    if k_part < parallel_k_parts - 1:
                        mask_c = mask(auto_copy(), [m - bid_x * bm, n - bid_y * bn])
                        tg_c = tensor_view(
                            c_partial[k_part, bid_x * bm : (bid_x + 1) * bm, bid_y * bn : (bid_y + 1) * bn], 
                            TensorLayout((bm, bn), (n, 1)),
                            "global",
                        )
                        # Copy results to global memory with masking
                        txrx_c = partition_src(tr_C, auto_copy())
                        txgx_c = partition_dst(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txrx_c, txgx_c, mask_c)
                        syncthreads()
                        if threadIdx.x == 0:
                            atomic_add(lc, 1, sem="acq_rel")
                    else:
                        acquire_seq_semaphore(lc, k_part)
                        # Write results back to global memory
                        mask_c = mask(auto_copy(), [m - bid_x * bm, n - bid_y * bn])
                        tr_c_partial = make_tensor("float32", auto_layout, "register")
                        for k_part_ in range(parallel_k_parts - 1):
                            tg_c = tensor_view(
                                c_partial[k_part_, bid_x * bm : (bid_x + 1) * bm, bid_y * bn : (bid_y + 1) * bn],
                                TensorLayout((bm, bn), (n, 1)),
                                "global",
                            )
                            # Copy results to global memory with masking
                            txrx_c = partition_dst(tr_c_partial, auto_copy())
                            txgx_c = partition_src(tg_c, auto_copy())
                            copy(auto_copy((bm, bn)), txgx_c, txrx_c, mask_c)
                            tr_C = tr_C + tr_c_partial
                        tg_c = tensor_view(c[bid_x * bm : (bid_x + 1) * bm, bid_y * bn : (bid_y + 1) * bn], 
                            TensorLayout((bm, bn), (n, 1)),
                            "global",
                        )
                        # Copy results to global memory with masking
                        txrx_c = partition_src(cast(tr_C, bf16), auto_copy())
                        txgx_c = partition_dst(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txrx_c, txgx_c, mask_c)

        return script_module.ir_module()


def w8a8_scaled_mm(n, k, group_n, group_k):
    scaled_mm_kernel = W8A8ScaledMM(n, k, group_n, group_k)
    return scaled_mm_kernel


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
    from vllm.model_executor.layers.quantization.utils.hidet_scaled_mm import w8a8_scaled_mm
    from vllm.model_executor.layers.quantization.utils.hidet_scaled_mm import per_token_group_quant_fp8_hidet_impl
    w8a8_scaled_mm_kernel = w8a8_scaled_mm(n, k, group_n, group_k)
    a, b, scale_a, scale_b, c = f8_quant_data(
        m, n, k, trans_b=True, return_hidet=True, group_m=1, group_n=group_n, group_k=group_k
    )

    def fn():
        return w8a8_scaled_mm_kernel(a, b, scale_a, scale_b)

    mean = do_bench(fn, percentiles=None)
    flops = 2.0 * m * n * k
    memory = f8e4m3.nbytes * (m * k + k * n) + f16.nbytes * m * n
    print("Hexcute: time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    hexcute_mean = mean

    c = w8a8_scaled_mm_kernel(a, b, scale_a, scale_b)

    print("c:", c)
    torch_a = a.torch()
    torch_b = b.torch()
    torch_scale_a = scale_a.torch().transpose(1, 0).contiguous()
    torch_scale_b = scale_b.torch()

    try:
        from vllm import _custom_ops as ops

        cutlass_scaled_fp8_gemm = ops.cutlass_scaled_mm
        cutlass_scaled_fp8_gemm(torch_a, torch_b.T, torch_scale_a.T, torch_scale_b.T, out_dtype=torch.bfloat16)
    except (ImportError, ValueError):
        cutlass_scaled_fp8_gemm = None

    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_block_fp8_matmul

        vllm_scaled_fp8_gemm = w8a8_block_fp8_matmul
        vllm_scaled_fp8_gemm(
            torch_a, torch_b, torch_scale_a.T, torch_scale_b, block_size=[group_k, group_k], output_dtype=torch.bfloat16
        )
    except (ImportError, ValueError):
        vllm_scaled_fp8_gemm = None

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
            actual=c.to(torch.float32).cpu().numpy(), desired=c2.to(torch.float32).cpu().numpy(), rtol=1e-2, atol=1e-2
        )
        if vllm_scaled_fp8_gemm is not None:
            c3 = fn3()
            np.testing.assert_allclose(
                actual=c2.to(torch.float32).cpu().numpy(), desired=c3.to(torch.float32).cpu().numpy(), rtol=1e-2, atol=1e-2
            )
            print("c:", c)
            print("c2:", c2)
            print("c3:", c3)
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
#    hidet.option.num_local_workers(1)

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
    headers = ["mxnxk", "triton", "cutlass", "hexcute"]
    records = []

    for m in [1, 2, 7, 8, 16, 32, 64, 100, 128, 3200]:
    #for m in [128]:
        #for m in [32, 64, 128, 2048, 4096]:
        for n, k in [[5120, 2048], [2048, 4096]]:
            #    for n, k in weight_shapes:
            time_hexcute, time_cutlass, time_triton = main(m, n, k, group_m, group_n, group_k, args.cand)
            shape = f"{m}x{n}x{k}"
            records.append([shape, time_triton, time_cutlass, time_hexcute])

    with open(args.output, "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )


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
