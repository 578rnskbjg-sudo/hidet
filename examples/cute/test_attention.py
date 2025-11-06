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
)
from quant_utils import canonicalize, bench

from hidet.utils import initialize

from hidet.ir.library import tune


_tiled_mma_pairs: List[Tuple[TiledMma, TiledMma]] = []


@initialize()
def register_tiled_mma():
    for head_size in [32, 64, 96, 128, 160, 192, 224, 256]:
        for warp in [2, 4, 8]:
            for repeat_m in [1, 2]:
                for repeat_n in [2, 4, 8]:
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
    def __init__(
        self, batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k
    ):
        self.batch_size = batch_size
        self.seqlen_q = seqlen_q
        self.num_heads = num_heads
        self.head_size = head_size
        self.seqlen_k = seqlen_k
        self.num_heads_k = num_heads_k

    def modules(self):
        return tune.extract_ir_modules(self._candidates)

    @tune.space(
        2,
        tiled_mma_pairs=_tiled_mma_pairs,
    )
    def _candidates(self, tiled_mma_pairs: Tuple[TiledMma, TiledMma]):
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
        tune.check(dynamic_smem_bytes <= 99000)

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
                tr_v = make_tensor(
                    "float16", layout_auto((head_size, inst_n * 2)), "register"
                )
                tr_qk = make_tensor("float32", auto_layout, "register")
                tr_o = make_tensor("float32", auto_layout, "register")
                tr_qk_max = make_tensor(
                    "float32", layout_auto((bm, bn), (1, 0)), "register"
                )
                tr_qk_sum = make_tensor(
                    "float32", layout_auto((bm, bn), (1, 0)), "register"
                )
                fill(tr_o, 0.0)
                fill(tr_qk_max, -float_max)
                fill(tr_qk_sum, 0.0)

                tg_q = tensor_view(
                    q[batch_idx, pid_m * bm :, head_idx, :],
                    TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                    "global",
                )
                txgq = partition_src(tg_q, auto_copy())
                tg_k = tensor_view(
                    k[batch_idx, 0:, head_idx, :],
                    TensorLayout((bn, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgk = partition_src(tg_k, auto_copy())
                tg_v = tensor_view(
                    v[batch_idx, 0:, head_idx, :],
                    TensorLayout((head_size, seqlen_k), (1, num_heads_k * head_size)),
                    "global",
                )
                txgv = partition_src(tg_v, auto_copy())

                ts_q = make_tensor("float16", TensorLayout((bm, head_size), (head_size, 1)), "shared")
                txsq = partition_dst(ts_q, auto_copy())
                ts_k = make_tensor("float16", TensorLayout((bn, head_size), (head_size, 1)), "shared")
                txsk = partition_dst(ts_k, auto_copy())
                ts_v = make_tensor("float16", layout_auto((head_size, bn), (1, head_size)), "shared")
                txsv = partition_dst(ts_v, auto_copy())

                copy(auto_copy((bm, head_size)), txgq, txsq)
                copy(auto_copy((bn, head_size)), txgk, txsk)
                cp_async_commit_group()

                txSq = partition_src(ts_q, auto_copy())
                txrq = partition_dst(tr_q, auto_copy())

                txSk = partition_src(ts_k, auto_copy())
                txrk = partition_dst(tr_k, auto_copy())

                txSv = partition_src(ts_v, auto_copy())
                txrv = partition_dst(tr_v, auto_copy())

                cp_async_wait_group(0)
                syncthreads()
                copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])
                copy(auto_copy(), txSk[:, :, 0], txrk[:, :, 0])

                h_tile_max = (head_size + inst_h - 1) // inst_h
                n_tile_max = (bn + inst_n - 1) // inst_n
                no_size = (seqlen_k + bn - 1) // bn
                for no in range(no_size):
                    fill(tr_qk, 0.0)

                    cp_async_wait_group(0)
                    syncthreads()

                    if no >= 1:
                        copy(auto_copy(), txSk[:, :, 0], txrk[:, :, 0])

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
                                txSk[:, :, h_tile_next],
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
                    scale = exp2((tr_qk_max - elementwise_max(tr_qk1_max, tr_qk_max)) * sm_scale)
                    tr_qk_max = elementwise_max(tr_qk1_max, tr_qk_max)
                    tr_qk_exp = exp2(tr_qk * sm_scale - tr_qk_max * sm_scale)
                    tr_qk1_sum = reduce_sum(tr_qk_exp, axis=1)
                    scale1 = broadcast_to(scale, tr_o)
                    tr_o = tr_o * scale1
                    tr_qk_sum = tr_qk_sum * scale + tr_qk1_sum
                    tr_qk_f16 = cast(tr_qk_exp, f16)
                    tr_qk1_f16 = partition_A(tr_qk_f16, tiled_mma_o)
                    cp_async_wait_group(0)
                    syncthreads()

                    if no < no_size - 1:
                        tg_k1 = tensor_view(
                            k[batch_idx, (no + 1) * bn :, head_idx, :],
                            TensorLayout((bn, head_size), (num_heads_k * head_size, 1)),
                            "global",
                        )
                        txgk1 = partition_src(tg_k1, auto_copy())
                        copy(auto_copy((bn, head_size)), txgk1[:, :], txsk[:, :])
                    cp_async_commit_group()

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


def flash_attention_v2_fwd(
    batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k
):
    flashattn = FlashAttention(
        batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k
    )
    return flashattn.modules()


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
    o = torch.empty(
        (batch_size, seqlen_q, num_heads, head_size), dtype=dtype, device=device
    )

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
def test_flash_attention_v2(
    batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k
):
    func = flash_attention_v2_fwd(
        batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k
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
    print(
        "time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean))
    )
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    def fn():
        func(q, k, v, o)

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print(
        "time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean))
    )

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
    print(
        "time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean))
    )

    o2 = fn()
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(
        actual=o.cpu().numpy(), desired=o2.cpu().numpy(), rtol=1e-2
    )


if __name__ == "__main__":
    hidet.option.cache_dir("attn")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k = (
        1,
        16,
        16,
        64,
        8192,
        8192,
    )
    print(
        f"batch_size: {batch_size}, num_heads: {num_heads}, num_heads_k: {num_heads_k}, head_size: {head_size}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}"
    )
    modules = flash_attention_v2_fwd(
        batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k
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

    for i, m in enumerate(modules):
        func = m.build()

        tiled_mma_qk, tiled_mma_o = m._tuning_kwargs["tiled_mma_pairs"]
        print(tiled_mma_qk.str_indented())
        print(tiled_mma_o.str_indented())

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
        print(
            "time={:.3f} ms, performance={:.3f} TFLOPS".format(
                mean, flops / (1e9 * mean)
            )
        )
        print(
            "time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean))
        )

        if best_time is None:
            best_time = mean
            best_i = i
            best_module = m
        elif mean < best_time:
            best_time = mean
            best_i = i
            best_module = m

    print(
        f"batch_size: {batch_size}, num_heads: {num_heads}, num_heads_k: {num_heads_k}, head_size: {head_size}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}"
    )
    print(best_i)
    print(best_time)

    func = best_module.build()
    func(q, k, v, o)

    from flash_attn.flash_attn_interface import flash_attn_func
    q = q.torch()
    k = k.torch()
    v = v.torch()
    out = flash_attn_func(q, k, v, causal=False)

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(
        actual=o.cpu().numpy(), desired=out.cpu().numpy(), rtol=1e-2
    )
    print(o.shape)
    print(out.shape)

    def fn():
        flash_attn_func(q, k, v, causal=False)

    mean, min_lat, max_lat = bench(fn, ())
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
    print(
        "time={:.3f} ms, performance={:.3f} TFLOPS".format(
            mean, flops / (1e9 * mean)
        )
    )
    print(
        "time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean))
    )
