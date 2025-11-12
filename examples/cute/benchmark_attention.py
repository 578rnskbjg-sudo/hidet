from typing import Tuple, List
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
    elementwise_max,
    broadcast_to,
    fill,
    transpose,
)
from quant_utils import canonicalize, bench

from hidet.utils import initialize

from hidet.ir.library import tune


_tiled_mma_pairs: List[Tuple[TiledMma, TiledMma]] = []


@initialize()
def register_tiled_mma():
    for head_size in [32, 64, 96, 128, 160, 192, 224, 256]:
        for warp in [4, 8]:#, 8]:
            for repeat_m in [1, 2]:#1, 2]:
                for repeat_n in [2, 4, 8]:#[2, 4, 8]:
                    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
                    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
                    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
                    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
                    warp_in_threadblock = Level(
                        "warp",
                        "thread_block",
                        (warp, 1),
                        TensorLayout((warp, 1)),
                        (repeat_m, repeat_n),
                    )
                    tiled_mma_qk = TiledMma(mma_atom, [warp_in_threadblock])

                    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
                    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
                    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
                    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
                    warp_in_threadblock = Level(
                        "warp",
                        "thread_block",
                        (warp, 1),
                        TensorLayout((warp, 1)),
                        (repeat_m, head_size // 16),
                    )
                    tiled_mma_o = TiledMma(mma_atom, [warp_in_threadblock])
                    _tiled_mma_pairs.append((tiled_mma_qk, tiled_mma_o))


class FlashAttention:
    def __init__(self, batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, sm_ver=80):
        self.batch_size = batch_size
        self.seqlen_q = seqlen_q
        self.num_heads = num_heads
        self.head_size = head_size
        self.seqlen_k = seqlen_k
        self.num_heads_k = num_heads_k
        self.sm_ver = sm_ver

    def modules(self):
        return tune.extract_ir_modules(self._pingpong) + tune.extract_ir_modules(self._pipeline)

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
        2,
        tiled_mma_pairs=_tiled_mma_pairs,
    )
    def _pingpong(self, tiled_mma_pairs: Tuple[TiledMma, TiledMma]):
        tiled_mma_qk, tiled_mma_o = tiled_mma_pairs

        from hidet.lang.types import u32, i32, f16, f32
        from hidet.lang import attrs
        from hidet.lang.cuda import syncthreads, cp_async_wait_all
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

        sm_scale = (1.0 / head_size) ** 0.5 * 1.44269504  # log2(e)
        float_max = f32.max_value

        dynamic_smem_bytes = (bm * head_size + bn * head_size + bn * head_size) * f16.nbytes
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        max_smem = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(dynamic_smem_bytes <= max_smem)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                q: f16[batch_size, seqlen_q, num_heads, head_size],
                k: f16[batch_size, seqlen_k, num_heads_k, head_size],
                v: f16[batch_size, seqlen_k, num_heads_k, head_size],
                o: f16[batch_size, seqlen_q, num_heads, head_size],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = bs * cdiv(seqlen_q, bm), 1
                attrs.cuda.dynamic_smem_bytes = 0

                pid = blockIdx.x
                grid_m = cdiv(seqlen_q, bm)
                pid_m = pid % grid_m
                bs_idx = pid // grid_m
                batch_idx = bs_idx // num_heads
                head_idx = bs_idx % num_heads

                tr_q = make_tensor("float16", layout_auto((bm, inst_h * 2)), "register")
                tr_k = make_tensor("float16", layout_auto((bn, inst_h * 2)), "register")
                tr_o = make_tensor("float32", auto_layout, "register")
                tr_qk_sum = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                tr_qk_max = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                fill(tr_o, 0.0)
                fill(tr_qk_sum, 0.0)
                fill(tr_qk_max, -float_max)

                tg_q = tensor_view(
                    q[batch_idx, pid_m * bm :, head_idx, :],
                    TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                    "global",
                )
                txgq = partition_src(tg_q, auto_copy())
                tg_k = tensor_view(
                    k[batch_idx, 0:, head_idx, :],
                    TensorLayout((head_size, seqlen_k), (1, num_heads_k * head_size)),
                    "global",
                )
                txgk = partition_src(tg_k, auto_copy())
                tg_v = tensor_view(
                    v[batch_idx, 0:, head_idx, :],
                    TensorLayout((head_size, seqlen_k), (1, num_heads_k * head_size)),
                    "global",
                )
                txgv = partition_src(tg_v, auto_copy())

                ts_q = make_tensor(
                    "float16", TensorLayout((bm, head_size), (head_size, 1)), "shared"
                )
                txsq = partition_dst(ts_q, auto_copy())
                ts_k = make_tensor(
                    "float16", TensorLayout((head_size, bn), (1, head_size)), "shared"
                )
                txsk = partition_dst(ts_k, auto_copy())

                ts_v = make_tensor(
                    "float16", TensorLayout((head_size, bn), (1, head_size)), "shared"
                )
                txsv = partition_dst(ts_v, auto_copy())

                copy(auto_copy((bm, head_size)), txgq, txsq)
                copy(auto_copy((head_size, bn)), txgk[:, :, 0], txsk)
                cp_async_commit_group()

                txSq = partition_src(ts_q, auto_copy())

                ts_kt = transpose(ts_k, 1, 0)
                txSkt = partition_src(ts_kt, auto_copy())
                txSv = partition_src(ts_v, auto_copy())

                cp_async_wait_group(0)
                syncthreads()

                txrq = partition_dst(tr_q, auto_copy())
                txrk = partition_dst(tr_k, auto_copy())

                copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])
                copy(auto_copy(), txSkt[:, :, 0], txrk[:, :, 0])

                h_tile_max = (head_size + inst_h - 1) // inst_h
                n_tile_max = (bn + inst_n - 1) // inst_n
                no_size = (seqlen_k + bn - 1) // bn
                for no in grid(no_size):
                    tr_qk = make_tensor("float32", auto_layout, "register")
                    fill(tr_qk, 0.0)

                    cp_async_wait_group(0)
                    syncthreads()

                    if no >= 1:
                        copy(auto_copy(), txSkt[:, :, 0], txrk[:, :, 0])

                    copy(auto_copy((head_size, bn)), txgv[:, :, no], txsv)
                    cp_async_commit_group()

                    for hi in grid(h_tile_max, attrs="u"):
                        h_tile_next = (hi + 1) % h_tile_max
                        copy(
                            auto_copy(),
                            txSq[:, :, h_tile_next],
                            txrq[:, :, (hi + 1) % 2],
                        )
                        if hi < h_tile_max - 1:
                            copy(
                                auto_copy(),
                                txSkt[:, :, h_tile_next],
                                txrk[:, :, (hi + 1) % 2],
                            )
                        mma(
                            tiled_mma_qk,
                            tr_qk,
                            txrq[:, :, hi % 2],
                            txrk[:, :, hi % 2],
                            tr_qk,
                        )

                    cp_async_wait_group(0)
 
                    if no < no_size - 1:
                        copy(auto_copy((head_size, bn)), txgk[:, :, no + 1], txsk[:, :])
                    cp_async_commit_group()
                   
                    tr_qk1_max = reduce_max(tr_qk, axis=1)
                    tr_qk_max_new = elementwise_max(tr_qk1_max, tr_qk_max)
                    scale = exp2((tr_qk_max - tr_qk_max_new) * sm_scale)
                    tr_qk_exp = exp2(tr_qk * sm_scale - tr_qk_max_new * sm_scale)
                    scale1 = broadcast_to(scale, tr_o)
                    tr_qk1_sum = reduce_sum(tr_qk_exp, axis=1)
                    tr_qk_sum = tr_qk_sum * scale + tr_qk1_sum
                    tr_qk_max = tr_qk_max_new
                    tr_qk_f16 = cast(tr_qk_exp, f16)
                    tr_qk1_f16 = partition_A(tr_qk_f16, tiled_mma_o)
                    tr_o = tr_o * scale1
                    syncthreads()
                    
                    tr_v = make_tensor("float16", layout_auto((head_size, inst_n * 2)), "register")
                    txrv = partition_dst(tr_v, auto_copy())
                    copy(auto_copy(), txSv[:, :, 0], txrv[:, :, 0])

                    for ni in grid(n_tile_max, attrs="u"):
                        if ni < n_tile_max - 1:
                            copy(
                                auto_copy(),
                                txSv[:, :, ni + 1],
                                txrv[:, :, (ni + 1) % 2],
                            )
                        mma(
                            tiled_mma_o,
                            tr_o,
                            tr_qk1_f16[:, :, ni],
                            txrv[:, :, ni % 2],
                            tr_o,
                        )

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

    @tune.space(
        2,
        tiled_mma_pairs=_tiled_mma_pairs,
        stages=[2, 3, 4]
    )
    def _pipeline(self, tiled_mma_pairs: Tuple[TiledMma, TiledMma], stages: int):
        tiled_mma_qk, tiled_mma_o = tiled_mma_pairs

        from hidet.lang.types import u32, i32, f16, f32
        from hidet.lang import attrs
        from hidet.lang.cuda import syncthreads, cp_async_wait_all
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

        sm_scale = (1.0 / head_size) ** 0.5 * 1.44269504  # log2(e)
        float_max = f32.max_value

        dynamic_smem_bytes = bm * head_size * f16.nbytes + (bn * head_size + bn * head_size) * f16.nbytes * stages
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        max_smem = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(dynamic_smem_bytes <= max_smem)
        unroll = f"u{stages}"

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                q: f16[batch_size, seqlen_q, num_heads, head_size],
                k: f16[batch_size, seqlen_k, num_heads_k, head_size],
                v: f16[batch_size, seqlen_k, num_heads_k, head_size],
                o: f16[batch_size, seqlen_q, num_heads, head_size],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = bs * cdiv(seqlen_q, bm), 1
                attrs.cuda.dynamic_smem_bytes = 0

                pid = blockIdx.x
                grid_m = cdiv(seqlen_q, bm)
                pid_m = pid % grid_m
                bs_idx = pid // grid_m
                batch_idx = bs_idx // num_heads
                head_idx = bs_idx % num_heads

                tr_q = make_tensor("float16", layout_auto((bm, inst_h * 2)), "register")
                tr_k = make_tensor("float16", layout_auto((bn, inst_h * 2)), "register")
                tr_o = make_tensor("float32", auto_layout, "register")
                tr_qk_sum = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                tr_qk_max = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                fill(tr_o, 0.0)
                fill(tr_qk_sum, 0.0)
                fill(tr_qk_max, -float_max)

                tg_q = tensor_view(
                    q[batch_idx, pid_m * bm :, head_idx, :],
                    TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                    "global",
                )
                txgq = partition_src(tg_q, auto_copy())
                tg_k = tensor_view(
                    k[batch_idx, 0:, head_idx, :],
                    TensorLayout((head_size, seqlen_k), (1, num_heads_k * head_size)),
                    "global",
                )
                txgk = partition_src(tg_k, auto_copy())
                tg_v = tensor_view(
                    v[batch_idx, 0:, head_idx, :],
                    TensorLayout((head_size, seqlen_k), (1, num_heads_k * head_size)),
                    "global",
                )
                txgv = partition_src(tg_v, auto_copy())

                ts_q = make_tensor(
                    "float16", TensorLayout((bm, head_size), (head_size, 1)), "shared"
                )
                txsq = partition_dst(ts_q, auto_copy())
                ts_k = make_tensor(
                    "float16", TensorLayout((head_size, bn, stages), (1, head_size, bn * head_size)), "shared"
                )
                txsk = partition_dst(ts_k, auto_copy())

                ts_v = make_tensor(
                    "float16", TensorLayout((head_size, bn, stages), (1, head_size, bn * head_size)), "shared"
                )
                txsv = partition_dst(ts_v, auto_copy())

                copy(auto_copy((bm, head_size)), txgq, txsq)
                cp_async_commit_group()
                cp_async_wait_group(0)
               
                for s in range(stages - 1):
                    copy(auto_copy((head_size, bn)), txgk[:, :, s], txsk[:, :, s])
                    copy(auto_copy((head_size, bn)), txgv[:, :, s], txsv[:, :, s])
                    cp_async_commit_group()

                txSq = partition_src(ts_q, auto_copy())

                ts_kt = transpose(ts_k, 1, 0, 2)
                txSkt = partition_src(ts_kt, auto_copy())
                txSv = partition_src(ts_v, auto_copy())

                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()

                txrq = partition_dst(tr_q, auto_copy())
                txrk = partition_dst(tr_k, auto_copy())

                copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])
                copy(auto_copy(), txSkt[:, :, 0, 0], txrk[:, :, 0])

                h_tile_max = (head_size + inst_h - 1) // inst_h
                n_tile_max = (bn + inst_n - 1) // inst_n
                no_size = (seqlen_k + bn - 1) // bn
                smem_pipe_read = 0
                smem_pipe_write = stages - 1
                for no in grid(no_size, unroll):
                    if no + stages - 1 < no_size:
                        copy(auto_copy((head_size, bn)), txgk[:, :, no + stages - 1], txsk[:, :, smem_pipe_write])
                        copy(auto_copy((head_size, bn)), txgv[:, :, no + stages - 1], txsv[:, :, smem_pipe_write])
                    cp_async_commit_group()
                    tr_qk = make_tensor("float32", auto_layout, "register")
                    fill(tr_qk, 0.0)

                    if no >= 1:
                        copy(auto_copy(), txSkt[:, :, 0, smem_pipe_read], txrk[:, :, 0])

                    for hi in grid(h_tile_max, attrs="u"):
                        h_tile_next = (hi + 1) % h_tile_max
                        copy(
                            auto_copy(),
                            txSq[:, :, h_tile_next],
                            txrq[:, :, (hi + 1) % 2],
                        )
                        if hi < h_tile_max - 1:
                            copy(
                                auto_copy(),
                                txSkt[:, :, h_tile_next, smem_pipe_read],
                                txrk[:, :, (hi + 1) % 2],
                            )
                        mma(
                            tiled_mma_qk,
                            tr_qk,
                            txrq[:, :, hi % 2],
                            txrk[:, :, hi % 2],
                            tr_qk,
                        )

                    tr_qk1_max = reduce_max(tr_qk, axis=1)
                    tr_qk_max_new = elementwise_max(tr_qk1_max, tr_qk_max)
                    scale = exp2((tr_qk_max - tr_qk_max_new) * sm_scale)
                    tr_qk_exp = exp2(tr_qk * sm_scale - tr_qk_max_new * sm_scale)
                    scale1 = broadcast_to(scale, tr_o)
                    tr_qk1_sum = reduce_sum(tr_qk_exp, axis=1)
                    tr_qk_sum = tr_qk_sum * scale + tr_qk1_sum
                    tr_qk_max = tr_qk_max_new
                    tr_qk_f16 = cast(tr_qk_exp, f16)
                    tr_qk1_f16 = partition_A(tr_qk_f16, tiled_mma_o)
                    tr_o = tr_o * scale1
                    
                    #tr_qk1_max = reduce_max(tr_qk, axis=1)
                    #scale = exp2((tr_qk_max - elementwise_max(tr_qk1_max, tr_qk_max)) * sm_scale)
                    #tr_qk_max = elementwise_max(tr_qk1_max, tr_qk_max)
                    #tr_qk_exp = exp2(tr_qk * sm_scale - tr_qk_max * sm_scale)
                    #scale1 = broadcast_to(scale, tr_o)
                    #tr_qk1_sum = reduce_sum(tr_qk_exp, axis=1)
                    #tr_qk_sum = tr_qk_sum * scale + tr_qk1_sum
                    #tr_qk_f16 = cast(tr_qk_exp, f16)
                    #tr_qk1_f16 = partition_A(tr_qk_f16, tiled_mma_o)
                    #tr_o = tr_o * scale1

                    tr_v = make_tensor("float16", layout_auto((head_size, inst_n * 2)), "register")
                    txrv = partition_dst(tr_v, auto_copy())
                    copy(auto_copy(), txSv[:, :, 0, smem_pipe_read], txrv[:, :, 0])

                    for ni in grid(n_tile_max, attrs="u"):
                        if ni < n_tile_max - 1:
                            copy(
                                auto_copy(),
                                txSv[:, :, ni + 1, smem_pipe_read],
                                txrv[:, :, (ni + 1) % 2],
                            )
                        mma(
                            tiled_mma_o,
                            tr_o,
                            tr_qk1_f16[:, :, ni],
                            txrv[:, :, ni % 2],
                            tr_o,
                        )
                
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read += 1
                    if smem_pipe_read == stages:
                        smem_pipe_read = 0
                    cp_async_wait_group(allow_on_fly_groups=stages - 2)
                    syncthreads()
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


def flash_attention_v2_fwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, sm_ver):
    flashattn = FlashAttention(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, sm_ver)
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
    q = torch.randint(
        low=lo,
        high=hi,
        size=(batch_size, seqlen_q, num_heads, head_size),
        dtype=dtype,
        device=device,
    )
    k = torch.randint(
        low=lo,
        high=hi,
        size=(batch_size, seqlen_k, num_heads_k, head_size),
        dtype=dtype,
        device=device,
    )
    v = torch.randint(
        low=lo,
        high=hi,
        size=(batch_size, seqlen_k, num_heads_k, head_size),
        dtype=dtype,
        device=device,
    )
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
    [
        (1, 16, 16, 128, 1024, 1024),
        (1, 16, 16, 128, 2048, 2048),
        (1, 16, 16, 128, 4096, 4096),
    ],
)
def test_flash_attention_v2(batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k):
    func = flash_attention_v2_fwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k)
    q, k, v, o = data(
        batch_size,
        seqlen_q,
        num_heads,
        head_size,
        seqlen_k,
        num_heads_k,
        return_hidet=True,
    )
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
    np.testing.assert_allclose(actual=o.cpu().numpy(), desired=o2.cpu().numpy(), rtol=1e-2)


def main(batch_sizes: int, num_heads: int, num_heads_k: int, head_size: int, seqlen_q: int, seqlen_k: int, sm_ver: int, cand: int = None):
    artifacts = flash_attention_v2_fwd(
        batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, sm_ver
    )
    q, k, v, o = data(
        batch_size,
        seqlen_q,
        num_heads,
        head_size,
        seqlen_k,
        num_heads_k,
        return_hidet=True,
    )

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
        stages = m._tuning_kwargs.get("stages", None)
        if stages is not None:
            print(f"stages={stages}")

        mean, min_lat, max_lat = bench(func, (q, k, v, o))
        def fn():
            func(q, k, v, o)
        mean = do_bench(fn, percentiles=None)

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
    mean_hexcute = best_time

    func = best_func
    func(q, k, v, o)

    from flash_attn.flash_attn_interface import flash_attn_func

    q = q.torch()
    k = k.torch()
    v = v.torch()
    out = flash_attn_func(q, k, v, causal=False)

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=o.cpu().numpy(), desired=out.cpu().numpy(), rtol=1e-2)
    print(o.shape)
    print(out.shape)

    def fn():
        flash_attn_func(q, k, v, causal=False)

    mean, min_lat, max_lat = bench(fn, ())
    mean = do_bench(fn, percentiles=None)
    mean_flash_atten = mean
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

    from flash_attention import attention as triton_attention
    q = q.permute(0, 2, 1, 3).contiguous() 
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()

    def fn3():
        return triton_attention(q, k, v, False, (1.0 / head_size) ** 0.5)
    out_triton = fn3()
    out_triton = out_triton.permute(0, 2, 1, 3)

    mean = do_bench(fn3, percentiles=None)
    # mean, min_lat, max_lat = bench(fn3, ())
    mean_triton = mean
    print("triton: time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    print("triton: time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
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
        "--search-space",
        type=int,
        choices=[1, 2],
        default=2,
        help="Search space of Hidet, can be either 1 or 2",
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache for generated kernels")
    parser.add_argument("--cand", "-i", type=int, default=None, help="The candidate you want to pick")
    parser.add_argument("--arch", type=int, default=None, help="Compute capability of CUDA")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="whether enabling debug mode or not"
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="output txt")

    #hidet.option.num_local_workers(1)
    args = parser.parse_args()
    if args.cache_dir is not None:
        hidet.option.cache_dir(args.cache_dir)
    hidet.option.search_space(args.search_space)
    if args.debug:
        hidet.option.debug_cache_tuning()
        hidet.option.save_lower_ir(True)

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
        mean_hexcute, mean_triton, mean_flash_atten = main(batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k, sm_ver, args.cand)
        exit()

    from tabulate import tabulate

    records = []
    headers = ["batch_size", "num_heads", "num_heads_k", "head_size", "seqlen_q", "seqlen_k", "hexcute", "triton", "flash-atten"]
    records = []
    hexcute = []
    triton = []
    flashattn = []

    for batch_size in [1]:
        for num_heads in [16, 32]:
            for head_size in [64, 128]:
                for seqlen_q in [512, 1024, 2048, 4096, 16384]:
                    num_heads_k = num_heads
                    seqlen_k = seqlen_q
                    flops = 2.0 * (
                        batch_size * seqlen_q * num_heads * seqlen_k * head_size
                        + batch_size * seqlen_q * num_heads_k * seqlen_k * head_size
                    )
                    mean_hexcute, mean_triton, mean_flash_atten = main(batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k, sm_ver)
                    flops_hexcute = flops / mean_hexcute / 1e9
                    flops_triton = flops / mean_triton / 1e9
                    flops_flash_atten = flops / mean_flash_atten / 1e9
                    records.append([batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k, mean_hexcute, mean_triton, mean_flash_atten])
                    hexcute.append(flops_hexcute)
                    triton.append(flops_triton)
                    flashattn.append(flops_flash_atten)

    with open(args.output, "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )
    
    methods = ['Hexcute', 'Marlin-old', 'Marlin-new', 'Triton', 'Ladder', 'cuBLAS']
    methods = ['Hexcute', 'FlashAttention', 'FlashInfer', 'Triton', 'CUTLASS', 'cuBLAS']
    
    clist = ['#b5739d', '#7ea6e0', '#67ab9f', '#ea6b66', '#ffb570', '#97d077']
    #clist = ['#38761D', '#4285F4', '#EA4335', '#ea6b66', '#ffb570', '#97d077']
    my_colors = {}
    for i, method in enumerate(methods):
        my_colors[method] = clist[i]

    import matplotlib.pyplot as plt
    from matplotlib import rc     
    rc('font', **{'family': 'sans-serif', 'size': 25})
    import numpy as np

    # Data for each method
    methods = ['Triton', 'FlashAttention', 'Hexcute']
    
    fig, ax = plt.subplots(1, 1, figsize=(30, 4))

    print(len(triton))
    print(len(flashattn))
    print(len(hexcute))
    categories = [f"M{m}" for m in range(len(flashattn))]
#    categories = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, '2K', '4K', '8K', '16K']
    # categories = [1, 8, 16, 32, 64, 128, 256]
    N = len(categories)
    ind = np.arange(N)  # X locations for the groups
    width = 0.2         # Width of the bars

    import numpy as np
    cmap = plt.get_cmap('gnuplot')
    ll = cmap.N*8//9
    len_methods = len(methods)
    indices = np.linspace(ll//5, ll, len_methods)
    #my_colors = [cmap(int(i)) for i in indices]
 
    # Plotting the bars for each method across matrix types (speedup)
#    print(len(speedup_marlin), len(speedup_tri), len(speedup_hi))
    gap = 0.012
    i = 0
    ax.bar(ind + (i + 0.5) * (width + gap), triton, width, label=methods[i], color=my_colors[methods[i]])
    i = 1
    ax.bar(ind + (i + 0.5) * (width + gap), flashattn, width, label=methods[i] + '2', color=my_colors[methods[i]])
    i = 2
    ax.bar(ind + (i + 0.5) * (width + gap), hexcute, width, label=methods[i], color=my_colors[methods[i]])
 
    #v = ind[-1] + 2.5 * (width + 0.2) * 0.6
    #ax.text(v, speedup_hi[-1], f'{1 / speedup_tri[-1]:.2f}x', ha='center', va='bottom', fontsize=16)
    #ax.axhline(y=1, color='b', linestyle='--', linewidth=2)

 #   x = marlin_old
 #   for i in range(len(marlin_old)):
 #       if x[i] >= 14:
 #           ax.text(ind[i] + 0.5 * (width + gap), 14, f'{marlin_old[i]:.0f}', ha='center', va='bottom', fontsize=10, color='black')

 #   x = triton
 #   for i in range(len(marlin_old)):
 #       if x[i] >= 14:
 #           ax.text(ind[i] + (1 + 0.5) * (width + gap), 13, f'{x[i]:.0f}', ha='center', va='bottom', fontsize=10, color='black')

    ax.set_ylabel('Throughput (TFLOPS)', fontsize=18)
    ax.set_ylim(0, 220)
    ax.set_xlabel('Fused Multi-head Attention Forward Layers', fontsize=18)
    ax.set_xticks(ind + (len(methods) * width) / 2)
    ax.set_yticks(np.arange(0, 220, 20))
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
    plt.savefig(args.output.replace(".txt", ".pdf"), dpi=300, bbox_inches="tight")
