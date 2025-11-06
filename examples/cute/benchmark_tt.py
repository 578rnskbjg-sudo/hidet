import torch
import pytest

import hidet
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
    fill,
    cast,
)
from hidet.lang.mapping import auto_map


@torch.no_grad()
def bench(functor, args):
    warmup_iters = 10
    bench_iters = 400

    for _ in range(warmup_iters):
        functor(*args)

    latencies = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(bench_iters):
        functor(*args)

    end.record()
    end.synchronize()
    latencies.append(start.elapsed_time(end) / bench_iters)
    print(latencies)

    mean = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    return mean, min_lat, max_lat


def data(Z, V, W, Y, U, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -2
    hi = 2
    a = torch.randint(low=lo, high=hi, size=(Z, V, W), dtype=dtype, device=device)
    # a = torch.ones(M, K, dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=(Y, V, U), dtype=dtype, device=device)
    # b = torch.ones(K, N, dtype=dtype, device=device)
    c = torch.empty((Z, Y, W, U), dtype=dtype, device=device)

    if return_hidet:
        a = hidet.from_torch(a)
        b = hidet.from_torch(b)
        c = hidet.from_torch(c)

    return a, b, c


def gemm_multi_buffer(Z, V, W, Y, U):
    from hidet.lang.types import u32, i32, f64, f16
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (2, 4))
    warp_in_threadblock = Level("warp", "thread_block", (2, 4), TensorLayout((2, 4)), (2, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    #a = TensorLayout(((4, 8), (1, )), ((8, 1), (8, )))
    #b = TensorLayout(((4, 8), (1, )), ((8, 1), (8, )))
    #c = TensorLayout(((4, 8), (2, )), ((16, 1), (8, )))
    #mma_atom = MmaAtom("warp", (8, 8, 4), a, b, c, c, (1, 1))
    #warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (4, 4))
    #tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    a_shape, a_tv_layout = tiled_mma.a_tv_layout()
    b_shape, b_tv_layout = tiled_mma.b_tv_layout()
    c_shape, c_tv_layout = tiled_mma.c_tv_layout()

    bm, inst_k = a_shape
    bn, inst_k_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_
    bk = 32

    def canonicalize(layout: TensorLayout):
        return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))

    a_t, a_v = canonicalize(a_tv_layout)
    b_t, b_v = canonicalize(b_tv_layout)
    c_t, c_v = canonicalize(c_tv_layout)

    threads = c_t.size()

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    stages = 3

    from hidet.utils.py import cdiv
    from hidet.lang import grid

    m = Z*W
    n = Y*U
    k = V
    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[Z, V, W], b: f16[Y, V, U], c: f16[Z, Y, W, U]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            group_size_m = 8
            pid = blockIdx.x
            num_pid_m = cdiv(m, bm)
            num_pid_n = cdiv(n, bn)
            num_pid_in_group = group_size_m * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * group_size_m
            group_size_m = min(num_pid_m - first_pid_m, group_size_m)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            ts_a = make_tensor("float16", layout_auto((bm, bk, stages)), "shared")
            ts_b = make_tensor("float16", layout_auto((bn, bk, stages)), "shared")

            tr_a = make_tensor("float16", layout_auto((bm, inst_k * 2)), "register")
            tr_b = make_tensor("float16", layout_auto((bn, inst_k * 2)), "register")
            tr_c = make_tensor("float16", auto_layout, "register")
            fill(tr_c, 0.0)

            tg_a = tensor_view(a[pid_m * bm//W :, :, :], TensorLayout(((bm//W, W), k), ((V*W, 1), W)), "global")
            tg_b = tensor_view(b[pid_n * bn//U:, :, :], TensorLayout(((bn//U, U), k), ((V*U, 1), U)), "global")

            txga = partition_src(tg_a, auto_copy())
            txsa = partition_dst(ts_a, auto_copy())

            txgb = partition_src(tg_b, auto_copy())
            txsb = partition_dst(ts_b, auto_copy())

            for s in range(stages - 1):
                copy(auto_copy((bm, bk)), txga[:, :, s], txsa[:, :, s])
                copy(auto_copy((bn, bk)), txgb[:, :, s], txsb[:, :, s])
                cp_async_commit_group()
            cp_async_wait_group(allow_on_fly_groups=stages - 2)
            syncthreads()

            smem_pipe_read = 0
            smem_pipe_write = stages - 1

            txSa = partition_src(ts_a, auto_copy())
            txra = partition_dst(tr_a, auto_copy())

            txSb = partition_src(ts_b, auto_copy())
            txrb = partition_dst(tr_b, auto_copy())

            txSa_p = txSa[:, :, :, smem_pipe_read]
            txSb_p = txSb[:, :, :, smem_pipe_read]

            copy(auto_copy(), txSa_p[:, :, 0], txra[:, :, 0])
            copy(auto_copy(), txSb_p[:, :, 0], txrb[:, :, 0])

            k_tile_max = bk // inst_k
            for ko in range((k + bk - 1) // bk):
                for ki in range(k_tile_max):
                    if ki == k_tile_max - 1:
                        cp_async_wait_group(allow_on_fly_groups=stages - 2)
                        syncthreads()

                    k_tile_next = (ki + 1) % k_tile_max
                    copy(auto_copy(), txSa[:, :, k_tile_next, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                    copy(auto_copy(), txSb[:, :, k_tile_next, smem_pipe_read], txrb[:, :, (ki + 1) % 2])
                    if ki == 0:
                        if ko + stages - 1 < (k + bk - 1) // bk:
                            copy(auto_copy((bm, bk)), txga[:, :, ko + stages - 1], txsa[:, :, smem_pipe_write])
                            copy(auto_copy((bn, bk)), txgb[:, :, ko + stages - 1], txsb[:, :, smem_pipe_write])
                        smem_pipe_write = smem_pipe_read
                        cp_async_commit_group()

                    if ki == k_tile_max - 2:
                        smem_pipe_read += 1
                        smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                    mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)

            tg_c = tensor_view(
                c[pid_m * bm//W : (pid_m + 1) * bm//W, pid_n * bn//U: (pid_n + 1) * bn//U, :, :],
                TensorLayout(((bm//W, W), (bn//U, U)), ((Y*W*U, U), (W*U, 1))),
                "global",
            )

            tr_C = rearrange(tr_c, auto_layout, "register")

            txrx_c = partition_src(tr_C, auto_copy())
            txgx_c = partition_dst(tg_c, auto_copy())
            copy(auto_copy((bm, bn)), txrx_c, txgx_c)

    func = script_module.build()
    return func


#def gemm_single_buffer(m, n, k):
#    from hidet.lang.types import u32, i32, f16, f32
#    from hidet.lang import attrs
#    from hidet.lang import shared_tensor, register_tensor
#    from hidet.lang.cuda import syncthreads, cp_async_wait_all
#    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
#    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group
#
#    # a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
#    # b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
#    # c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
#    # mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 2))
#    # warp_in_threadblock = Level("warp", "thread_block", (3, 2), TensorLayout((3, 2)), (1, 2))
#    # tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
#
#    # block_m = 32
#    # block_n = 32
#    # m % block_m == 0 and n % block_n == 0
#    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
#    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
#    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
#    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
#    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (1, 1))
#    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
#
#    #    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
#    #    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
#    #    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
#    #    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (2, 8))
#    #    warp_in_threadblock = Level("warp", "thread_block", (2, 4), TensorLayout((2, 4)), (1, 1))
#    #    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
#
#    # a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
#    # b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
#    # c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
#    # mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 1))
#    # warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (2, 2))
#    # tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
#
#    a_shape, a_tv_layout = tiled_mma.a_tv_layout()
#    b_shape, b_tv_layout = tiled_mma.b_tv_layout()
#    c_shape, c_tv_layout = tiled_mma.c_tv_layout()
#
#    bm, inst_k = a_shape
#    bn, inst_k_ = b_shape
#    bm_, bn_ = c_shape
#    assert bm == bm_ and bn == bn_ and inst_k == inst_k_
#    bk = 32
#
#    def canonicalize(layout: TensorLayout):
#        return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))
#
#    a_t, a_v = canonicalize(a_tv_layout)
#    b_t, b_v = canonicalize(b_tv_layout)
#    c_t, c_v = canonicalize(c_tv_layout)
#
#    threads = c_t.size()
#
#    a_elements = a_v.size()
#    b_elements = b_v.size()
#    c_elements = c_v.size()
#
#    from hidet.ir.cute.algorithm import auto_copy, auto_mma
#    from hidet.ir.cute import auto_layout
#    from hidet.ir.cute import layout_auto
#
#    stages = 1
#    dynamic_smem_bytes = (bm + bn) * bk * stages * f16.nbytes
#
#    from hidet.utils.py import cdiv
#    from hidet.lang import grid
#    from hidet.ir.expr import cast
#
#    def convert(x):
#        return cast(x, f16)
#
#    with hidet.script_module() as script_module:
#
#        @hidet.script
#        def func(a: f16[m, k], b: f16[k, n], c: f16[m, n]):
#            attrs.func_kind = "cuda_kernel"
#            attrs.cuda.block_dim = threads
#            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
#            attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes
#
#            group_size_m = 8
#            pid = blockIdx.x
#            num_pid_m = cdiv(m, bm)
#            num_pid_n = cdiv(n, bn)
#            num_pid_in_group = group_size_m * num_pid_n
#            group_id = pid // num_pid_in_group
#            first_pid_m = group_id * group_size_m
#            group_size_m = min(num_pid_m - first_pid_m, group_size_m)
#            pid_m = first_pid_m + (pid % group_size_m)
#            pid_n = (pid % num_pid_in_group) // group_size_m
#
#            smem_a = dynamic_shared_memory(byte_offset=0, dtype=f16)
#            smem_b = dynamic_shared_memory(byte_offset=stages * bm * bk * f16.nbytes, dtype=f16)
#
#            a_regs = register_tensor("float16", shape=[a_elements, 2])
#            b_regs = register_tensor("float16", shape=[b_elements, 2])
#            c_regs = register_tensor("float32", shape=[c_elements])
#
#            for i in grid(c_elements):
#                c_regs[i] = 0.0
#
#            tg_a = tensor_view(a[pid_m * bm :, :], TensorLayout((bm, k), (k, 1)), "global")
#            tg_b = tensor_view(b[:, pid_n * bn :], TensorLayout((bn, k), (1, n)), "global")
#
#            ts_a = tensor_view(smem_a, TensorLayout((bm, bk), (bk, 1)), "shared")
#            ts_b = tensor_view(smem_b, TensorLayout((bn, bk), (1, bn)), "shared")
#
#            tr_a = tensor_view(a_regs, layout_auto((bm, inst_k * 2)), "register")
#            tr_b = tensor_view(b_regs, layout_auto((bn, inst_k * 2)), "register")
#            tr_c = tensor_view(c_regs, auto_layout, "register")
#
#            txga = partition_src(tg_a, auto_copy())
#            txsa = partition_dst(ts_a, auto_copy())
#
#            txgb = partition_src(tg_b, auto_copy())
#            txsb = partition_dst(ts_b, auto_copy())
#
#            txSa = partition_src(ts_a, auto_copy())
#            txra = partition_dst(tr_a, auto_copy())
#
#            txSb = partition_src(ts_b, auto_copy())
#            txrb = partition_dst(tr_b, auto_copy())
#
#            k_tile_max = bk // inst_k
#            for ko in range((k + bk - 1) // bk):
#                copy(auto_copy((bm, bk)), txga[:, :, ko], txsa)
#                copy(auto_copy((bn, bk)), txgb[:, :, ko], txsb)
#                cp_async_wait_all()
#                syncthreads()
#
#                copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])
#                copy(auto_copy(), txSb[:, :, 0], txrb[:, :, 0])
#
#                for ki in range(k_tile_max):
#                    if ki < k_tile_max - 1:
#                        copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
#                        copy(auto_copy(), txSb[:, :, ki + 1], txrb[:, :, (ki + 1) % 2])
#
#                    mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)
#                syncthreads()
#
#            tg_c = tensor_view(
#                c[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
#                TensorLayout((bm, bn), (n, 1)),
#                "global",
#            )
#
#            tr_c_f16 = arithmetic(tr_c, op=convert)
#
#            tr_C = rearrange(tr_c_f16, auto_layout, "register")
#
#            txrx_c = partition_src(tr_C, auto_copy())
#            txgx_c = partition_dst(tg_c, auto_copy())
#            copy(auto_copy((bm, bn)), txrx_c, txgx_c)
#
#    func = script_module.build()
#    # a_mem = hidet.empty([m, k], device="cuda")
#    # b_mem = hidet.empty([k, n], device="cuda")
#    # c_mem = hidet.empty([m, n], device="cuda")
#    # func(a_mem, b_mem, c_mem)
#    return func


#@pytest.mark.parametrize("m,n,k", [(4096, 4096, 4096), (120000, 320, 768)])
#def test_gemm_single_buffer(m, n, k):
#    # hidet.option.cache_dir("./demo_mma")
#    # hidet.option.search_space(2)
#    # hidet.option.debug_cache_tuning()
#    # hidet.option.save_lower_ir(True)
#
#    func = gemm_single_buffer(m, n, k)
#    a, b, c = data(m, n, k, return_hidet=True)
#    mean, min_lat, max_lat = bench(func, (a, b, c))
#    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
#
#    def fn():
#        func(a, b, c)
#
#    from hidet.utils.benchmark import do_bench
#
#    mean = do_bench(fn, percentiles=None)
#    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
#
#    a = a.torch()
#    b = b.torch()
#
#    def fn():
#        c = a @ b
#
#    from hidet.utils.benchmark import do_bench
#
#    mean, min_lat, max_lat = bench(fn, ())
#    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
#
#    c2 = a @ b
#    import numpy as np
#
#    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
#    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)
#

import time
import torch
import cupy as cp
#import cupyx.cutensor as cutensor


def test_tt(V, Z, W, Y, U):
    hidet.option.cache_dir("./demo_tt_f16")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    #hidet.option.save_lower_ir(True)

    m = Z * W
    k = V
    n = Y * U
    func = gemm_multi_buffer(Z, V, W, Y, U)
    a, b, c = data(Z, V, W, Y, U, dtype="float16", return_hidet=True)
    mean, min_lat, max_lat = bench(func, (a, b, c))
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    def fn():
        func(a, b, c)

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    a = a.torch()
    b = b.torch()

    def fn():
        c = torch.einsum('zvw,yvu->zywu', a, b)
        return c.contiguous()

    mean, min_lat, max_lat = bench(fn, ())
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    def fn2(a, b):
        a = a.permute(0, 2, 1)
        a = a.reshape(Z*W, V)
        b = b.permute(1, 0, 2)
        b = b.reshape(V, Y*U)
        c = a @ b
        c = c.reshape(Z, W, Y, U)
        c = c.permute(0, 2, 1, 3)
        return c.contiguous()

    options = {"triton.cudagraphs": False, "epilogue_fusion": True, "max_autotune": True}
    #compiled_func = torch.compile(fn2, options=options)
    compiled_func = torch.compile(fn2, backend="hidet", mode="max-autotune-no-cudagraphs")
    mean, min_lat, max_lat = bench(compiled_func, (a, b))
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
    c2 = compiled_func(a, b)
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-4)

    return
    cutensor_prep_time = 0

    a = cp.random.random((Z, V, W)).astype(cp.float16)
    b = cp.random.random((Y, V, U)).astype(cp.float16)
    c = cp.random.random((Z, Y, W, U)).astype(cp.float16)

    a = cp.clip(a, -0.01, 0.01)
    b = cp.clip(b, -0.01, 0.01)
    c = cp.clip(c, -0.01, 0.01)

    tic = time.perf_counter()
    alpha = 1.0
    beta = 0.0
    mode_a = ('z', 'v', 'w')
    mode_b = ('y', 'v', 'u')
    mode_c = ('z', 'y', 'w', 'u')
    abc = cp.empty((Z, Y, W, U)).astype(cp.float16)
    torch.cuda.synchronize()
    cutensor_prep_time += time.perf_counter() - tic
    print(f'cutensor.contraction prep (one time): {cutensor_prep_time * 1000:.3f} ms')

    def fn():
        cutensor.contraction(alpha, a, mode_a, b, mode_b, beta, abc, mode_c)

    mean, min_lat, max_lat = bench(fn, ())
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))


if __name__ == "__main__":
    #test_gemm_multi_buffer_f64(1024, 1024, 1024)
    #test_gemm_multi_buffer_f64(2048, 2048, 2048)
    test_tt(4096, 128, 32, 256, 16)
