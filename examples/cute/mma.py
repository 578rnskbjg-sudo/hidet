import hidet
import torch

from hidet.lang.types import f16, f32
from hidet.lang import attrs
from hidet.lang.cuda import blockIdx, threadIdx
from hidet.utils.py import cdiv


from hidet.ir.cute.algorithm import MmaAtom, TiledMma, auto_copy
from hidet.ir.cute.layout import TensorLayout, Level
from hidet.ir.cute import layout_auto, auto_layout

from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    copy,
    mma,
    rearrange,
    cast,
    fill,
)


def data(M, N, K, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -2
    hi = 2
    a = torch.randint(low=lo, high=hi, size=(M, K), dtype=dtype, device=device)
    # a = torch.ones(M, K, dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=(N, K), dtype=dtype, device=device)
    # b = torch.ones(K, N, dtype=dtype, device=device)
    c = torch.empty((M, N), dtype=dtype, device=device)

    if return_hidet:
        a = hidet.from_torch(a)
        b = hidet.from_torch(b)
        c = hidet.from_torch(c)

    return a, b, c


def gemm_example(m, n, k):
    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c)
    warp_in_threadblock = Level("warp", "thread_block", (2, 4), TensorLayout((2, 4)))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    a_shape, _ = tiled_mma.a_tv_layout()
    b_shape, _ = tiled_mma.b_tv_layout()
    c_shape, _ = tiled_mma.c_tv_layout()

    bm, bk = a_shape
    bn, bk_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and bk == bk_

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b: f16[n, k], c: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 256  # 8 warps
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            bid_x = blockIdx.x
            bid_y = blockIdx.y

            tg_a = tensor_view(a[bid_x * bm : (bid_x + 1) * bm, :], TensorLayout((bm, k), (k, 1)), "global")
            tg_b = tensor_view(b[bid_y * bn : (bid_y + 1) * bn, :], TensorLayout((bn, k), (k, 1)), "global")

            tr_a = make_tensor("float16", layout_auto((bm, bk)), "register")
            tr_b = make_tensor("float16", layout_auto((bn, bk)), "register")
            tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
            fill(tr_c, 0.0)

            txga = partition_src(tg_a, auto_copy())
            txra = partition_dst(tr_a, auto_copy())
            txgb = partition_src(tg_b, auto_copy())
            txrb = partition_dst(tr_b, auto_copy())

            for ki in range(cdiv(k, bk)):
                copy(auto_copy((bm, bk)), txga[:, :, ki], txra)
                copy(auto_copy((bn, bk)), txgb[:, :, ki], txrb)
                mma(tiled_mma, tr_c, txra, txrb, tr_c)

            tr_C = rearrange(cast(tr_c, f16), auto_layout, "register")

            tg_c = tensor_view(c[bid_x * bm : (bid_x + 1) * bm, bid_y * bn : (bid_y + 1) * bn], TensorLayout((bm, bn), (n, 1)), "global")
            txgc = partition_src(tg_c, auto_copy())
            txrc = partition_dst(tr_C, auto_copy())
            copy(auto_copy((bm, bn)), txrc, txgc)
    func = script_module.build()
    return func


def main():
    hidet.option.cache_dir("./demo_mma_2")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    m = 1024
    n = 1024
    k = 1024
    func = gemm_example(m, n, k)
    a, b, c = data(m, n, k, return_hidet=True)
    func(a, b, c)

    a = a.torch()
    b = b.torch()
    c2 = a @ b.T
    import numpy as np

    print(c)
    print(c2)
    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)


if __name__ == "__main__":
    main()
