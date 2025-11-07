from typing import Tuple, List
from hidet.ir.cute.contexts import warp_groups_consumer, warp_groups_producer
import hidet
import pytest
import torch
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout, make_layout, coalesce
from hidet.ir.cute.layout import ThrValAtom, Level
from hidet.ir.cute.algorithm import CopyAtom, TiledCopy, MmaAtom, TiledMma
from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    mask,
    copy,
    mma,
    sub_tensor,
    rearrange,
    arithmetic,
    cast,
    exp,
    exp2,
    reduce_sum,
    reduce_max,
    partition_A,
    partition_B,
    elementwise_max,
    broadcast_to,
    fill,
    transpose,
    make_mbarriers,
    mbarrier_arrive,
    mbarrier_try_wait,
    mbarrier_wait,
    wgmma_fence_operand,
)
from quant_utils import canonicalize, bench
from hidet.ir.primitives.cuda import barrier_sync, barrier_arrive

from hidet.utils import initialize

from hidet.ir.library import tune


_tiled_mma_pairs: List[Tuple[TiledMma, TiledMma]] = []


@initialize()
def register_tiled_mma():
    for head_size in [32, 64, 96, 128, 160, 192, 224, 256]:
        for warp_group in [1, 2]:# [1, 2]:#[1, 2]:
            for n in [32, 64, 128, 256]:
                a = TensorLayout(((128,), (64, 16)), ((0,), (1, 64)))
                b = TensorLayout(((128,), (n, 16)), ((0,), (1, n)))
                c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((128, 1, 16), (64, 8, 512)))
                mma_atom = MmaAtom("warp_group", (64, n, 16), a, b, c, c)
                wg_in_threadblock = Level(
                    "warp_group", "thread_block", (warp_group, 1), TensorLayout((warp_group, 1)), (1, 1)
                )
                tiled_mma_qk = TiledMma(mma_atom, [wg_in_threadblock])
 
                a = TensorLayout(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512)))
                b = TensorLayout(((128,), (head_size, 16)), ((0,), (1, head_size)))
                c = TensorLayout(((4, 8, 4), (2, 2, head_size // 8)), ((128, 1, 16), (64, 8, 512)))
                mma_atom = MmaAtom("warp_group", (64, head_size, 16), a, b, c, c)
                wg_in_threadblock = Level(
                    "warp_group", "thread_block", (warp_group, 1), TensorLayout((warp_group, 1)), (1, 1)
                )
                tiled_mma_o = TiledMma(mma_atom, [wg_in_threadblock])
                _tiled_mma_pairs.append((tiled_mma_qk, tiled_mma_o))


class FlashAttention3:
    def __init__(self, batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, sm_ver=80):
        self.batch_size = batch_size
        self.seqlen_q = seqlen_q
        self.num_heads = num_heads
        self.head_size = head_size
        self.seqlen_k = seqlen_k
        self.num_heads_k = num_heads_k
        self.sm_ver = sm_ver

    def modules(self):
        return tune.extract_ir_modules(self._cooperative) + tune.extract_ir_modules(self._pingpong)

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

    @tune.space(2, tiled_mma_pairs=_tiled_mma_pairs, stages=[2], cluster_m=[1, 2])
    def _cooperative(self, tiled_mma_pairs: Tuple[TiledMma, TiledMma], stages: int, cluster_m: int):
        tiled_mma_qk, tiled_mma_o = tiled_mma_pairs

        from hidet.lang.types import u32, i32, f16, f32
        from hidet.lang import attrs
        from hidet.lang.cuda import syncthreads, cp_async_wait_all
        from hidet.ir.primitives.cuda.wgmma import wgmma_fence, wgmma_commit_group, wgmma_wait_group
        from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
        from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

        from hidet.ir.cute.algorithm import auto_copy, auto_mma
        from hidet.ir.cute import auto_layout, layout_auto

        from hidet.utils.py import cdiv
        from hidet.lang import grid

        batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k = (
            self.batch_size,
            self.seqlen_q,
            self.num_heads,
            self.head_size,
            self.seqlen_k,
            self.num_heads_k,
        )
        compute_capability = self.sm_ver

        q_shape, q_tv_layout = tiled_mma_qk.a_tv_layout()
        k_shape, k_tv_layout = tiled_mma_qk.b_tv_layout()
        qk_shape, qk_tv_layout = tiled_mma_qk.c_tv_layout()

        _, q_v = canonicalize(q_tv_layout)
        _, k_v = canonicalize(k_tv_layout)
        qk_t, qk_v = canonicalize(qk_tv_layout)

        threads = qk_t.size()

        bm, inst_h = q_shape
        bn, inst_h_ = k_shape
        bm_, bn_ = qk_shape
        assert bm == bm_ and bn == bn_ and inst_h == inst_h_

        qk_shape, qk_tv_layout = tiled_mma_o.a_tv_layout()
        v_shape, v_tv_layout = tiled_mma_o.b_tv_layout()
        o_shape, o_tv_layout = tiled_mma_o.c_tv_layout()

        _, inst_n = qk_shape

        _, qk_v = canonicalize(qk_tv_layout)
        _, v_v = canonicalize(v_tv_layout)
        _, o_v = canonicalize(o_tv_layout)

        _, head_size_ = o_shape
        tune.check(head_size == head_size_)
        bs = batch_size * num_heads
        assert num_heads == num_heads_k
        #tune.check(threads == 256)

        num_consumer_threads = threads
        num_producer_threads = 128

        if num_consumer_threads == 128:
            producer_warpgroups = [1]
            consumer_warpgroups = [0]
            producer_regs = 56 
            consumer_regs = 256
        else:    
            producer_warpgroups = [2]
            consumer_warpgroups = [0, 1]
            producer_regs = 40
            consumer_regs = 232

        sm_scale = (1.0 / head_size) ** 0.5 * 1.44269504  # log2(e)
        float_max = f32.max_value

        dynamic_smem_bytes = (bm * head_size + bn * head_size * 2 * stages) * f16.nbytes
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        max_smem = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(dynamic_smem_bytes <= max_smem)
        tma_copy_tx = bn * head_size * f16.nbytes
        tma_copy_tx_v = tma_copy_tx
        unroll = f"u{stages}"

        WarpSchedulerBarrierWG1 = 1 
        WarpSchedulerBarrierWG2 = 2
        WarpSchedulerBarrierWG3 = 3 
        UseSchedulerBarrier = num_consumer_threads >= 256
        NumThreadsPerWarpGroup = 128

        cluster_n = 1
        cluster_layout = TensorLayout((cluster_m, cluster_n), (1, cluster_m))
        cluster_size = cluster_layout.size()
        
        with hidet.script_module() as script_module:

            @hidet.script
            def warp_scheduler_barrier_init():
                if UseSchedulerBarrier:
                    warpgroup_id = threadIdx.x // 128
                    if warpgroup_id == 0:
                        barrier_arrive(WarpSchedulerBarrierWG1, NumThreadsPerWarpGroup * 2, aligned=True) 

            @hidet.script
            def warp_scheduler_barrier_sync():
                if UseSchedulerBarrier:
                    warpgroup_id = threadIdx.x // 128
                    barrier_sync(WarpSchedulerBarrierWG1 + warpgroup_id, NumThreadsPerWarpGroup * 2, aligned=True)

            @hidet.script
            def warp_scheduler_barrier_arrive():
                if UseSchedulerBarrier:
                    warpgroup_id = threadIdx.x // 128
                    cur_WG = warpgroup_id
                    next_WG = 1 - cur_WG
                    barrier_arrive(WarpSchedulerBarrierWG1 + next_WG, NumThreadsPerWarpGroup * 2, aligned=True)

            @hidet.script
            def func(
                q: f16[batch_size, seqlen_q, num_heads, head_size],
                k: f16[batch_size, seqlen_k, num_heads_k, head_size],
                v: f16[batch_size, seqlen_k, num_heads_k, head_size],
                o: f16[batch_size, seqlen_q, num_heads, head_size],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = num_producer_threads + num_consumer_threads
                attrs.cuda.grid_dim = bs * cdiv(seqlen_q, bm), 1
                attrs.cuda.cluster_dim = cluster_size
                attrs.cuda.min_blocks = 1
                attrs.cuda.dynamic_smem_bytes = 0

                pid = blockIdx.x
                grid_m = cdiv(seqlen_q, bm)
                pid_m = pid % grid_m
                bs_idx = pid // grid_m
                batch_idx = bs_idx // num_heads
                head_idx = bs_idx % num_heads

                tg_q = tensor_view(
                    q[batch_idx, :, head_idx, :], #q[0, :, :, :],
                    TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                    "global",
                    (bm, head_size),
                    (pid_m * bm, 0),
                )
                tg_k = tensor_view(
                    k, #k[batch_idx, :, head_idx, :], #k[0, :, :, :],
                    TensorLayout((num_heads_k * head_size, batch_size * seqlen_k), (1, num_heads_k * head_size)),
                    "global",
                    (head_size, batch_size * seqlen_k),
                    (head_idx * head_size, batch_idx * seqlen_k),
                )
                tg_v = tensor_view(
                    v, #v[batch_idx, :, head_idx, :], #v[0, :, :, :],
                    TensorLayout((num_heads_k * head_size, batch_size * seqlen_k), (1, num_heads_k * head_size)),
                    "global",
                    (head_size, batch_size * seqlen_k),
                    (head_idx * head_size, batch_idx * seqlen_k),
                )

                ts_q = make_tensor("float16", layout_auto((bm, head_size)), "shared")
                ts_k = make_tensor("float16", layout_auto((head_size, bn, stages)), "shared")
                ts_v = make_tensor("float16", layout_auto((head_size, bn, stages)), "shared")

                mbar_tma = make_mbarriers(stages)
                mbar_mma = make_mbarriers(stages)
                mbar_tma_v = make_mbarriers(stages)
                mbar_mma_v = make_mbarriers(stages)

                with warp_groups_producer(producer_warpgroups, num_regs=producer_regs):
                    smem_pipe_write = 0
                    write_phase = True
                    no_size = (seqlen_k + bn - 1) // bn
                    txgk = partition_src(tg_k, auto_copy())
                    txgv = partition_src(tg_v, auto_copy())
                    txsk = partition_dst(ts_k, auto_copy())
                    txsv = partition_dst(ts_v, auto_copy())
                    mask_ = mask(auto_copy(), [i32(head_size), i32(bn)])
                    for no in grid(no_size, attrs=unroll):
                        if no >= stages:
                            mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)
                        copy(
                            auto_copy((head_size, bn)),
                            txgk[:, :, no],
                            txsk[:, :, smem_pipe_write],
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        mbarrier_arrive(mbar_tma[smem_pipe_write], tma_copy_tx)
                        if no >= stages:
                            mbarrier_wait(mbar_mma_v[smem_pipe_write], write_phase)
                        copy(
                            auto_copy((head_size, bn)),
                            txgv[:, :, no],
                            txsv[:, :, smem_pipe_write],
                            mask_=mask_,
                            mbarrier=mbar_tma_v[smem_pipe_write],
                        )
                        mbarrier_arrive(mbar_tma_v[smem_pipe_write], tma_copy_tx_v)
                        smem_pipe_write += 1
                        if smem_pipe_write == stages:
                            smem_pipe_write = 0
                            write_phase = not write_phase
                    for no in range(stages):
                        mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)
                        mbarrier_wait(mbar_mma_v[smem_pipe_write], write_phase)
                        smem_pipe_write += 1
                        if smem_pipe_write == stages:
                            smem_pipe_write = 0
                            write_phase = not write_phase

                with warp_groups_consumer(consumer_warpgroups, num_regs=consumer_regs):
                    smem_pipe_read = 0
                    read_phase = False
                    smem_pipe_release = 0
                    release_phase = False
                    txgq = partition_src(tg_q, auto_copy())
                    txsq = partition_dst(ts_q, auto_copy())
                    copy(auto_copy((bm, head_size)), txgq[:, :], txsq)
                    cp_async_commit_group()
                    cp_async_wait_group(0)
                    syncthreads()
                    tr_o = make_tensor("float32", auto_layout, "register")
                    tr_qk_sum = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                    tr_qk_max = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                    fill(tr_o, 0.0)
                    fill(tr_qk_sum, 0.0)
                    fill(tr_qk_max, -float_max)
                    scale = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                    warp_scheduler_barrier_init()
                    
                    tQKsQ = partition_A(ts_q, tiled_mma_qk)
                    ts_kt = transpose(ts_k, 1, 0, 2)
                    tQKsK = partition_B(ts_kt, tiled_mma_qk)
                    tOsV = partition_B(ts_v, tiled_mma_o)

                    h_tile_max = (head_size + inst_h - 1) // inst_h
                    n_tile_max = (bn + inst_n - 1) // inst_n
                    no_size = (seqlen_k + bn - 1) // bn
                    warp_scheduler_barrier_sync()

                    for no in grid(no_size, attrs=unroll):
                        tr_qk = make_tensor("float32", auto_layout, "register")
                        fill(tr_qk, 0.0)

                        mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                        wgmma_fence_operand(tr_qk)
                        wgmma_fence()
                        for hi in grid(h_tile_max, attrs="u+"):
                            mma(tiled_mma_qk, tr_qk, tQKsQ[:, :, hi], tQKsK[:, :, hi, smem_pipe_read], tr_qk, cluster_layout=cluster_layout)
                        wgmma_commit_group()
                        wgmma_fence_operand(tr_qk)
                        warp_scheduler_barrier_arrive()
                        wgmma_wait_group(0)
                        mbarrier_arrive(mbar_mma[smem_pipe_release])

                        tr_qk1_max = reduce_max(tr_qk, axis=1)
                        tr_qk_max_new = elementwise_max(tr_qk1_max, tr_qk_max)
                        scale = exp2((tr_qk_max - tr_qk_max_new) * sm_scale)
                        tr_qk_max = tr_qk_max_new
                        tr_qk_exp = exp2(tr_qk * sm_scale - tr_qk_max * sm_scale)
                        scale1 = broadcast_to(scale, tr_o)
                        tr_qk1_sum = reduce_sum(tr_qk_exp, axis=1)
                        tr_qk_sum = tr_qk_sum * scale + tr_qk1_sum
                        tr_qk_f16 = cast(tr_qk_exp, f16)
                        tr_qk1_f16 = partition_A(tr_qk_f16, tiled_mma_o)
                        tr_o = tr_o * scale1

                        mbarrier_wait(mbar_tma_v[smem_pipe_read], read_phase)
                        warp_scheduler_barrier_sync()
                        wgmma_fence_operand(tr_o)
                        wgmma_fence()
                        for ni in grid(n_tile_max, attrs="u+"):
                            mma(tiled_mma_o, tr_o, tr_qk1_f16[:, :, ni], tOsV[:, :, ni, smem_pipe_read], tr_o, cluster_layout=cluster_layout)
                        wgmma_commit_group()
                        wgmma_fence_operand(tr_o)
                        wgmma_wait_group(0)
                        mbarrier_arrive(mbar_mma_v[smem_pipe_release])
                        smem_pipe_read += 1
                        if smem_pipe_read == stages:
                            smem_pipe_read = 0
                            read_phase = not read_phase
                        smem_pipe_release += 1
                        if smem_pipe_release == stages:
                            smem_pipe_release = 0
                            release_phase = not release_phase

                    warp_scheduler_barrier_arrive()

                    tr_qk1_sum = broadcast_to(tr_qk_sum, tr_o)
                    tr_o = tr_o / tr_qk1_sum

                    tg_o = tensor_view(
                        o[batch_idx, pid_m * bm :, head_idx, 0:],
                        TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                        "global",
                    )

                    tr_o_f16 = cast(tr_o, f16)

                    tr_O = rearrange(tr_o_f16, auto_layout, "register")

                    txrx_o = partition_src(tr_O, auto_copy())
                    txgx_o = partition_dst(tg_o, auto_copy())
                    copy(auto_copy((bm, head_size)), txrx_o, txgx_o)

        return script_module.ir_module()

    @tune.space(2, tiled_mma_pairs=_tiled_mma_pairs, stages=[2], cluster_m=[1, 2])
    def _pingpong(self, tiled_mma_pairs: Tuple[TiledMma, TiledMma], stages: int, cluster_m: int):
        tiled_mma_qk, tiled_mma_o = tiled_mma_pairs

        from hidet.lang.types import u32, i32, f16, f32
        from hidet.lang import attrs
        from hidet.lang.cuda import syncthreads, cp_async_wait_all
        from hidet.ir.primitives.cuda.wgmma import wgmma_fence, wgmma_commit_group, wgmma_wait_group
        from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
        from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

        from hidet.ir.cute.algorithm import auto_copy, auto_mma
        from hidet.ir.cute import auto_layout, layout_auto

        from hidet.utils.py import cdiv
        from hidet.lang import grid

        batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k = (
            self.batch_size,
            self.seqlen_q,
            self.num_heads,
            self.head_size,
            self.seqlen_k,
            self.num_heads_k,
        )
        compute_capability = self.sm_ver

        q_shape, q_tv_layout = tiled_mma_qk.a_tv_layout()
        k_shape, k_tv_layout = tiled_mma_qk.b_tv_layout()
        qk_shape, qk_tv_layout = tiled_mma_qk.c_tv_layout()

        _, q_v = canonicalize(q_tv_layout)
        _, k_v = canonicalize(k_tv_layout)
        qk_t, qk_v = canonicalize(qk_tv_layout)

        threads = qk_t.size()

        bm, inst_h = q_shape
        bn, inst_h_ = k_shape
        bm_, bn_ = qk_shape
        assert bm == bm_ and bn == bn_ and inst_h == inst_h_

        qk_shape, qk_tv_layout = tiled_mma_o.a_tv_layout()
        v_shape, v_tv_layout = tiled_mma_o.b_tv_layout()
        o_shape, o_tv_layout = tiled_mma_o.c_tv_layout()

        _, inst_n = qk_shape

        _, qk_v = canonicalize(qk_tv_layout)
        _, v_v = canonicalize(v_tv_layout)
        _, o_v = canonicalize(o_tv_layout)

        _, head_size_ = o_shape
        tune.check(head_size == head_size_)
        bs = batch_size * num_heads
        assert num_heads == num_heads_k
        #tune.check(threads == 256)
    
        num_consumer_threads = threads
        num_producer_threads = 128

        if num_consumer_threads == 128:
            producer_warpgroups = [1]
            consumer_warpgroups = [0]
            producer_regs = 56 
            consumer_regs = 256
        else:    
            producer_warpgroups = [2]
            consumer_warpgroups = [0, 1]
            producer_regs = 40
            consumer_regs = 232

        sm_scale = (1.0 / head_size) ** 0.5 * 1.44269504  # log2(e)
        float_max = f32.max_value

        dynamic_smem_bytes = (bm * head_size + bn * head_size * 2 * stages) * f16.nbytes
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        max_smem = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(dynamic_smem_bytes <= max_smem)
        tma_copy_tx = bn * head_size * f16.nbytes
        tma_copy_tx_v = tma_copy_tx
        unroll = f"u{stages}"

        WarpSchedulerBarrierWG1 = 1 
        WarpSchedulerBarrierWG2 = 2
        WarpSchedulerBarrierWG3 = 3 
        UseSchedulerBarrier = num_consumer_threads >= 256
        NumThreadsPerWarpGroup = 128

        cluster_n = 1
        cluster_layout = TensorLayout((cluster_m, cluster_n), (1, cluster_m))
        cluster_size = cluster_layout.size()
 
        with hidet.script_module() as script_module:

            @hidet.script
            def warp_scheduler_barrier_init():
                if UseSchedulerBarrier:
                    warpgroup_id = threadIdx.x // 128
                    if warpgroup_id == 0:
                        barrier_arrive(WarpSchedulerBarrierWG1, NumThreadsPerWarpGroup * 2, aligned=True) 

            @hidet.script
            def warp_scheduler_barrier_sync():
                if UseSchedulerBarrier:
                    warpgroup_id = threadIdx.x // 128
                    barrier_sync(WarpSchedulerBarrierWG1 + warpgroup_id, NumThreadsPerWarpGroup * 2, aligned=True)

            @hidet.script
            def warp_scheduler_barrier_arrive():
                if UseSchedulerBarrier:
                    warpgroup_id = threadIdx.x // 128
                    cur_WG = warpgroup_id
                    next_WG = 1 - cur_WG
                    barrier_arrive(WarpSchedulerBarrierWG1 + next_WG, NumThreadsPerWarpGroup * 2, aligned=True)

            @hidet.script
            def func(
                q: f16[batch_size, seqlen_q, num_heads, head_size],
                k: f16[batch_size, seqlen_k, num_heads_k, head_size],
                v: f16[batch_size, seqlen_k, num_heads_k, head_size],
                o: f16[batch_size, seqlen_q, num_heads, head_size],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = num_producer_threads + num_consumer_threads
                attrs.cuda.grid_dim = bs * cdiv(seqlen_q, bm), 1
                attrs.cuda.cluster_dim = cluster_size
                attrs.cuda.min_blocks = 1
                attrs.cuda.dynamic_smem_bytes = 0

                pid = blockIdx.x
                grid_m = cdiv(seqlen_q, bm)
                pid_m = pid % grid_m
                bs_idx = pid // grid_m
                batch_idx = bs_idx // num_heads
                head_idx = bs_idx % num_heads

                tg_q = tensor_view(
                    q[batch_idx, :, head_idx, :], #q[0, :, :, :],
                    TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                    "global",
                    (bm, head_size),
                    (pid_m * bm, 0),
                )
                tg_k = tensor_view(
                    k, #k[batch_idx, :, head_idx, :], #k[0, :, :, :],
                    TensorLayout((num_heads_k * head_size, batch_size * seqlen_k), (1, num_heads_k * head_size)),
                    "global",
                    (head_size, batch_size * seqlen_k),
                    (head_idx * head_size, batch_idx * seqlen_k),
                )
                tg_v = tensor_view(
                    v, #v[batch_idx, :, head_idx, :], #v[0, :, :, :],
                    TensorLayout((num_heads_k * head_size, batch_size * seqlen_k), (1, num_heads_k * head_size)),
                    "global",
                    (head_size, batch_size * seqlen_k),
                    (head_idx * head_size, batch_idx * seqlen_k),
                )

                ts_q = make_tensor("float16", layout_auto((bm, head_size)), "shared")
                ts_k = make_tensor("float16", layout_auto((head_size, bn, stages)), "shared")
                ts_v = make_tensor("float16", layout_auto((head_size, bn, stages)), "shared")

                mbar_tma = make_mbarriers(stages)
                mbar_mma = make_mbarriers(stages)
                mbar_tma_v = make_mbarriers(stages)
                mbar_mma_v = make_mbarriers(stages)
                
                with warp_groups_producer(producer_warpgroups, num_regs=producer_regs):
                    smem_pipe_write = 0
                    write_phase = True
                    no_size = (seqlen_k + bn - 1) // bn
                    txgk = partition_src(tg_k, auto_copy())
                    txgv = partition_src(tg_v, auto_copy())
                    txsk = partition_dst(ts_k, auto_copy())
                    txsv = partition_dst(ts_v, auto_copy())
                    mask_ = mask(auto_copy(), [i32(head_size), i32(bn)])
                    for no in grid(no_size, attrs=unroll):
                        if no >= stages:
                            mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)
                        copy(
                            auto_copy((head_size, bn)),
                            txgk[:, :, no],
                            txsk[:, :, smem_pipe_write],
                            mbarrier=mbar_tma[smem_pipe_write],
                        )
                        mbarrier_arrive(mbar_tma[smem_pipe_write], tma_copy_tx)
                        if no >= stages:
                            mbarrier_wait(mbar_mma_v[smem_pipe_write], write_phase)
                        copy(
                            auto_copy((head_size, bn)),
                            txgv[:, :, no],
                            txsv[:, :, smem_pipe_write],
                            mask_=mask_,
                            mbarrier=mbar_tma_v[smem_pipe_write],
                        )
                        mbarrier_arrive(mbar_tma_v[smem_pipe_write], tma_copy_tx_v)
                        smem_pipe_write += 1
                        if smem_pipe_write == stages:
                            smem_pipe_write = 0
                            write_phase = not write_phase
                    for no in range(stages):
                        mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)
                        mbarrier_wait(mbar_mma_v[smem_pipe_write], write_phase)
                        smem_pipe_write += 1
                        if smem_pipe_write == stages:
                            smem_pipe_write = 0
                            write_phase = not write_phase

                with warp_groups_consumer(consumer_warpgroups, num_regs=consumer_regs):
                    smem_pipe_read = 0
                    read_phase = False
                    smem_pipe_read_v = 0
                    read_phase_v = False
                    smem_pipe_release = 0
                    smem_pipe_release_v = 0
                    txgq = partition_src(tg_q, auto_copy())
                    txsq = partition_dst(ts_q, auto_copy())
                    copy(auto_copy((bm, head_size)), txgq[:, :], txsq)
                    cp_async_commit_group()
                    cp_async_wait_group(0)
                    syncthreads()
                   
                    warp_scheduler_barrier_init()

                    tr_o = make_tensor("float32", auto_layout, "register")
                    tr_qk_sum = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                    tr_qk_max = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                    fill(tr_o, 0.0)
                    fill(tr_qk_sum, 0.0)
                    fill(tr_qk_max, -float_max)

                    tQKsQ = partition_A(ts_q, tiled_mma_qk)
                    ts_kt = transpose(ts_k, 1, 0, 2)
                    tQKsK = partition_B(ts_kt, tiled_mma_qk)
                    tOsV = partition_B(ts_v, tiled_mma_o)

                    h_tile_max = (head_size + inst_h - 1) // inst_h
                    n_tile_max = (bn + inst_n - 1) // inst_n
                    no_size = (seqlen_k + bn - 1) // bn
                    
                    tr_qk_f16 = make_tensor("float16", layout_auto((bm, bn)), "register")
                    tOrQK = partition_A(tr_qk_f16, tiled_mma_o)
                    
                    tr_qk = make_tensor("float32", auto_layout, "register")
                    fill(tr_qk, 0.0)
                    
                    mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                    wgmma_fence_operand(tr_qk)
                    wgmma_fence()
                    for hi in grid(h_tile_max, attrs="u+"):
                        mma(tiled_mma_qk, tr_qk, tQKsQ[:, :, hi], tQKsK[:, :, hi, smem_pipe_read], tr_qk, cluster_layout=cluster_layout)
                    wgmma_commit_group()
                    wgmma_fence_operand(tr_qk)
                    wgmma_wait_group(0)
                    mbarrier_arrive(mbar_mma[smem_pipe_release])
                    smem_pipe_read += 1
                    if smem_pipe_read == stages:
                        smem_pipe_read = 0
                        read_phase = not read_phase
                    smem_pipe_release += 1
                    if smem_pipe_release == stages:
                        smem_pipe_release = 0
                    
                    tr_qk1_max = reduce_max(tr_qk, axis=1)
                    tr_qk_max_new = elementwise_max(tr_qk1_max, tr_qk_max)
                    scale = exp2((tr_qk_max - tr_qk_max_new) * sm_scale)
                    tr_qk_max = tr_qk_max_new
                    tr_qk_exp = exp2(tr_qk * sm_scale - tr_qk_max * sm_scale)
                    tr_qk1_sum = reduce_sum(tr_qk_exp, axis=1)
                    tr_qk_sum = tr_qk_sum * scale + tr_qk1_sum
                    tr_qk_f16 = cast(tr_qk_exp, f16)

                    for no in grid(no_size - 1, attrs=unroll):
                        fill(tr_qk, 0.0)

                        mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                        warp_scheduler_barrier_sync()
                        wgmma_fence_operand(tr_qk)
                        wgmma_fence()
                        for hi in grid(h_tile_max, attrs="u+"):
                            mma(tiled_mma_qk, tr_qk, tQKsQ[:, :, hi], tQKsK[:, :, hi, smem_pipe_read], tr_qk, cluster_layout=cluster_layout)
                        wgmma_commit_group()
                        wgmma_fence_operand(tr_qk)
                        
                        scale1 = broadcast_to(scale, tr_o)
                        tr_o = tr_o * scale1
                        mbarrier_wait(mbar_tma_v[smem_pipe_read_v], read_phase_v)
                        wgmma_fence_operand(tr_o)
                        wgmma_fence()
                        for ni in grid(n_tile_max, attrs="u+"):
                            mma(tiled_mma_o, tr_o, tOrQK[:, :, ni], tOsV[:, :, ni, smem_pipe_read_v], tr_o, cluster_layout=cluster_layout)
                        wgmma_commit_group()
                        wgmma_fence_operand(tr_o)
                        warp_scheduler_barrier_arrive()
                        wgmma_wait_group(1)
                        mbarrier_arrive(mbar_mma[smem_pipe_release])
 
                        tr_qk1_max = reduce_max(tr_qk, axis=1)
                        tr_qk_max_new = elementwise_max(tr_qk1_max, tr_qk_max)
                        scale = exp2((tr_qk_max - tr_qk_max_new) * sm_scale)
                        tr_qk_max = tr_qk_max_new
                        tr_qk_exp = exp2(tr_qk * sm_scale - tr_qk_max * sm_scale)
                        tr_qk1_sum = reduce_sum(tr_qk_exp, axis=1)
                        tr_qk_sum = tr_qk_sum * scale + tr_qk1_sum
                        tr_qk_f16 = cast(tr_qk_exp, f16)

                        wgmma_wait_group(0)
                        mbarrier_arrive(mbar_mma_v[smem_pipe_release_v])

                        smem_pipe_read += 1
                        if smem_pipe_read == stages:
                            smem_pipe_read = 0
                            read_phase = not read_phase
                        smem_pipe_read_v += 1
                        if smem_pipe_read_v == stages:
                            smem_pipe_read_v = 0
                            read_phase_v = not read_phase_v
                        smem_pipe_release += 1
                        if smem_pipe_release == stages:
                            smem_pipe_release = 0
                        smem_pipe_release_v += 1
                        if smem_pipe_release_v == stages:
                            smem_pipe_release_v = 0

                    mbarrier_wait(mbar_tma_v[smem_pipe_read_v], read_phase_v)
                    wgmma_fence_operand(tr_o)
                    wgmma_fence()
                    for ni in grid(n_tile_max, attrs="u+"):
                        mma(tiled_mma_o, tr_o, tOrQK[:, :, ni], tOsV[:, :, ni, smem_pipe_read_v], tr_o, cluster_layout=cluster_layout)
                    wgmma_commit_group()
                    wgmma_fence_operand(tr_o)
                    wgmma_wait_group(0)
                    mbarrier_arrive(mbar_mma_v[smem_pipe_release_v])
               
                    scale1 = broadcast_to(scale, tr_o)
                    tr_o = tr_o * scale1
                    tr_qk2_sum = broadcast_to(tr_qk_sum, tr_o)
                    tr_o = tr_o / tr_qk2_sum

                    tg_o = tensor_view(
                        o[batch_idx, pid_m * bm :, head_idx, 0:],
                        TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                        "global",
                    )

                    tr_o_f16 = cast(tr_o, f16)

                    tr_O = rearrange(tr_o_f16, auto_layout, "register")

                    txrx_o = partition_src(tr_O, auto_copy())
                    txgx_o = partition_dst(tg_o, auto_copy())
                    copy(auto_copy((bm, head_size)), txrx_o, txgx_o)

        return script_module.ir_module()



def flash_attention_v3_fwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, sm_ver):
    flashattn = FlashAttention3(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, sm_ver)
    return flashattn.build()


def data(
    batch_size,
    seqlen_q,
    num_heads,
    head_size,
    seqlen_k,
    num_heads_k,
    dtype="float16",
    device="cuda",
    return_hidet=False,
):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    q = torch.randint(low=lo, high=hi, size=(batch_size, seqlen_q, num_heads, head_size), dtype=dtype, device=device)
    k = torch.randint(low=lo, high=hi, size=(batch_size, seqlen_k, num_heads_k, head_size), dtype=dtype, device=device)
    v = torch.randint(low=lo, high=hi, size=(batch_size, seqlen_k, num_heads_k, head_size), dtype=dtype, device=device)
    o = torch.empty((batch_size, seqlen_q, num_heads, head_size), dtype=dtype, device=device)

    q = q
    k = k / head_size
    v = v
    if return_hidet:
        q = hidet.from_torch(q)
        k = hidet.from_torch(k)
        v = hidet.from_torch(v)
        o = hidet.from_torch(o)

    return q, k, v, o


@pytest.mark.parametrize(
    "batch_size,num_heads,num_heads_k,head_size,seqlen_q,seqlen_k",
    [(1, 16, 16, 128, 1024, 1024), (1, 16, 16, 128, 2048, 2048), (1, 16, 16, 128, 4096, 4096)],
)
def test_flash_attention_v2(batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k):
    func = flash_attention_v2_fwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k)
    q, k, v, o = data(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, return_hidet=True)
    mean, min_lat, max_lat = bench(func, (q, k, v, o))
    flops = 2.0 * (
        batch_size * seqlen_q * num_heads * seqlen_k * head_size
        + batch_size * seqlen_q * num_heads_k * seqlen_k * head_size
    )
    from hidet.ir.dtypes import f16

    memory = f16.nbytes * (
        batch_size * num_heads * seqlen_q * head_size
        + batch_size * num_heads_k * seqlen_k * head_size
        + batch_size * num_heads_k * seqlen_k * head_size
        + batch_size * seqlen_q * num_heads * head_size
    )
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    def fn():
        func(q, k, v, o)

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))

    q = q.torch()
    k = k.torch()
    v = v.torch()
    o = o.torch()

    softmax = torch.nn.Softmax(dim=1)

    def fn():
        q1 = q.permute(0, 2, 1, 3)
        q2 = q1.view(batch_size * num_heads, seqlen_q, head_size)
        k1 = k.permute(0, 2, 3, 1)
        k2 = k1.view(batch_size * num_heads_k, head_size, seqlen_k)
        qk = q2 @ k2
        v1 = v.permute(0, 2, 1, 3)
        v2 = v1.view(batch_size * num_heads_k, seqlen_k, head_size)
        qk = qk.view(batch_size * num_heads * seqlen_q, seqlen_k)
        qk = softmax(qk.to(torch.float32))
        qk = qk.to(torch.float16)
        qk = qk.view(batch_size * num_heads, seqlen_q, seqlen_k)
        o1 = qk @ v2
        o2 = o1.view(batch_size, num_heads, seqlen_q, head_size)
        o3 = o2.permute(0, 2, 1, 3)
        return o3.contiguous()

    from hidet.utils.benchmark import do_bench

    mean, min_lat, max_lat = bench(fn, ())
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))

    o2 = fn()
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=o.cpu().numpy(), desired=o2.cpu().numpy(), rtol=1e-2, atol=0.5)


def main(
    batch_sizes: int,
    num_heads: int,
    num_heads_k: int,
    head_size: int,
    seqlen_q: int,
    seqlen_k: int,
    sm_ver: int,
    cand: int = None,
):
    artifacts = flash_attention_v3_fwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, sm_ver)
    q, k, v, o = data(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, return_hidet=True)

    best_time = None
    best_i = None
    best_module = None
    best_func = None

    from hidet.utils.benchmark import do_bench
    
    for i, (m, func) in enumerate(artifacts):
        if cand is not None and i != cand:
            continue

        tiled_mma_qk, tiled_mma_o = m._tuning_kwargs["tiled_mma_pairs"]
        print(tiled_mma_qk.str_indented())
        print(tiled_mma_o.str_indented())

        def fn():
            func(q, k, v, o)
        mean, min_lat, max_lat = bench(func, (q, k, v, o))
        flops = 2.0 * (
            batch_size * seqlen_q * num_heads * seqlen_k * head_size
            + batch_size * seqlen_q * num_heads_k * seqlen_k * head_size
        )
        from hidet.ir.dtypes import f16

        memory = f16.nbytes * (
            batch_size * num_heads * seqlen_q * head_size
            + batch_size * num_heads_k * seqlen_k * head_size
            + batch_size * num_heads_k * seqlen_k * head_size
            + batch_size * seqlen_q * num_heads * head_size
        )
        print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
        print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

        if best_time is None:
            best_time = mean
            best_i = i
            best_module = m
            best_func = func
        elif mean < best_time:
            best_time = mean
            best_i = i
            best_module = m
            best_func = func

    print(
        f"batch_size: {batch_size}, num_heads: {num_heads}, num_heads_k: {num_heads_k}, head_size: {head_size}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}"
    )
    print(best_i)
    print(best_time)
    func = best_func
    from hashlib import sha256
    mod = best_module
    hash_dir = sha256(str(mod).encode()).hexdigest()[:16]
    print(hash_dir)         

    def fn():
        func(q, k, v, o)
    mean, min_lat, max_lat = bench(func, (q, k, v, o))
    mean_hexcute = mean
    flops = 2.0 * (
        batch_size * seqlen_q * num_heads * seqlen_k * head_size
        + batch_size * seqlen_q * num_heads_k * seqlen_k * head_size
    )
    from hidet.ir.dtypes import f16

    memory = f16.nbytes * (
          batch_size * num_heads * seqlen_q * head_size
        + batch_size * num_heads_k * seqlen_k * head_size
        + batch_size * num_heads_k * seqlen_k * head_size
        + batch_size * seqlen_q * num_heads * head_size
    )
    print("Hexcute: time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    print("Hexcute: time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    func(q, k, v, o)

    from flash_attn_interface import flash_attn_func

    q = q.torch()
    k = k.torch()
    v = v.torch()

    def fn():
        return flash_attn_func(q, k, v, causal=False)

    out = fn()
    
    mean, min_lat, max_lat = bench(fn, ())
    mean_flash_atten = mean
    print("flash3: time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    print("flash3: time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    from flash_attention import attention as triton_attention
    q = q.permute(0, 2, 1, 3).contiguous() 
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()

    def fn3():
        return triton_attention(q, k, v, False, (1.0 / head_size) ** 0.5)
    out_triton = fn3()
    out_triton = out_triton.permute(0, 2, 1, 3)

    mean, min_lat, max_lat = bench(fn3, ())
    mean_triton = mean
    print("triton: time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    print("triton: time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=o.cpu().numpy(), desired=out.cpu().numpy(), rtol=5e-2, atol=0.5)
    np.testing.assert_allclose(actual=out.cpu().numpy(), desired=out_triton.cpu().numpy(), rtol=5e-2, atol=0.5)

    return mean_hexcute, mean_triton, mean_flash_atten


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark the performance of the standard attention layer provided by CUTE and flash-attention"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-heads-k", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--seqlen-q", type=int, default=4096)
    parser.add_argument("--seqlen-k", type=int, default=4096)
    parser.add_argument(
        "--search-space", type=int, choices=[1, 2], default=2, help="Search space of Hidet, can be either 1 or 2"
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache for generated kernels")
    parser.add_argument("--cand", "-i", type=int, default=None, help="The candidate you want to pick")
    parser.add_argument("--arch", type=int, default=None, help="Compute capability of CUDA")
    parser.add_argument("--debug", "-d", action="store_true", help="whether enabling debug mode or not")
    parser.add_argument("--output", "-o", type=str, default=None, help="output txt")

#    hidet.option.num_local_workers(1)
    args = parser.parse_args()
    if args.cache_dir is not None:
        hidet.option.cache_dir(args.cache_dir)
    hidet.option.search_space(args.search_space)
    if args.debug:
        hidet.option.debug_cache_tuning()
        hidet.option.save_lower_ir(True)
    hidet.option.use_torch_stream(True)

    batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k = (
        args.batch_size,
        args.num_heads,
        args.num_heads_k,
        args.head_size,
        args.seqlen_q,
        args.seqlen_k,
    )
    if args.arch is not None:
        sm_ver = args.arch
    else:
        sm_ver = hidet.option.cuda.get_arch_pair()
        sm_ver = sm_ver[0] * 10 + sm_ver[1]
    print(f"CUDA compute capability: {sm_ver}")
    if args.output is None:
        mean_hexcute, mean_triton, mean_flash_atten = main(
            batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k, sm_ver, args.cand
        )
        exit()

    from tabulate import tabulate

    records = []
    headers = ["batch_size", "num_heads", "num_heads_k", "head_size", "seqlen_q", "seqlen_k", "hexcute", "triton", "flash-atten"]
    triton = []
    flashattn = []
    hexcute = []

    for batch_size in [1]:
        for num_heads in [16, 32]:
            for head_size in [64, 128]:
                for seqlen_q in [512, 1024, 2048, 4096, 16384]:
                    num_heads_k = num_heads
                    seqlen_k = seqlen_q
                    mean_hexcute, mean_triton, mean_flash_atten = main(
                        batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k, sm_ver
                    )
                    flops = 2.0 * (
                        batch_size * seqlen_q * num_heads * seqlen_k * head_size
                        + batch_size * seqlen_q * num_heads_k * seqlen_k * head_size
                    )
                    flops_hexcute = flops / mean_hexcute / 1e9
                    flops_triton = flops / mean_triton / 1e9
                    flops_flash_atten = flops / mean_flash_atten / 1e9
                    records.append(
                        [
                            batch_size,
                            num_heads,
                            num_heads_k,
                            head_size,
                            seqlen_q,
                            seqlen_k,
                            mean_hexcute,
                            mean_triton,
                            mean_flash_atten,
                        ]
                    )
                    hexcute.append(flops_hexcute)
                    triton.append(flops_triton)
                    flashattn.append(flops_flash_atten)

    with open(args.output, "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )

    import matplotlib.pyplot as plt
    from matplotlib import rc     
    rc('font', **{'family': 'sans-serif', 'size': 25})
    import numpy as np
    methods = ['Hexcute', 'FlashAttention', 'FlashInfer', 'Triton', 'CUTLASS', 'cuBLAS']
    
    clist = ['#b5739d', '#7ea6e0', '#67ab9f', '#ea6b66', '#ffb570', '#97d077']
    #clist = ['#38761D', '#4285F4', '#EA4335', '#ea6b66', '#ffb570', '#97d077']
    my_colors = {}
    for i, method in enumerate(methods):
        my_colors[method] = clist[i]

    # Data for each method
    methods = ['Triton', 'FlashAttention', 'Hexcute']

    fig, ax = plt.subplots(1, 1, figsize=(30, 4))

    print(len(triton))
    print(len(flashattn))
    print(len(hexcute))
    categories = [f"M{m}" for m in range(len(flashattn))]
    N = len(categories)
    ind = np.arange(N)  # X locations for the groups
    width = 0.2         # Width of the bars

    import numpy as np
    cmap = plt.get_cmap('gnuplot')
    ll = cmap.N*8//9
    len_methods = len(methods)
    indices = np.linspace(ll//5, ll, len_methods)
    
    gap = 0.012
    i = 0
    ax.bar(ind + (i + 0.5) * (width + gap), triton, width, label=methods[i], color=my_colors[methods[i]])
    i = 1
    ax.bar(ind + (i + 0.5) * (width + gap), flashattn, width, label=methods[i] + '3', color=my_colors[methods[i]])
    i = 2
    ax.bar(ind + (i + 0.5) * (width + gap), hexcute, width, label=methods[i], color=my_colors[methods[i]])
 
    ax.set_ylabel('Throughput (TFLOPS)', fontsize=18)
    ax.set_ylim(0, 450)
    ax.set_xlabel('Fused Multi-head Attention Forward Layers', fontsize=18)
    ax.set_xticks(ind + (len(methods) * width) / 2)
    ax.set_yticks(np.arange(0, 450, 50))
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
    fig.legend(lins, labs, loc='upper left', bbox_to_anchor=(0.056, 0.95), fontsize=14, ncols=3)

    fig.subplots_adjust(
            top=0.94,
            bottom=0.173,
            left=0.056,
            right=0.99,
            hspace=0.2,
            wspace=0.2
        )
    # Adjust layout to prevent clipping of tick-labels
    plt.savefig(args.output.replace('.txt', '.pdf'), dpi=300, bbox_inches='tight')
