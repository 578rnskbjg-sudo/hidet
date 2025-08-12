from typing import Union
import hidet
import torch
import pytest
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout, make_layout, coalesce
from hidet.ir.cute.layout import ThrValAtom, Level
from hidet.ir.cute.algorithm import CopyAtom, TiledCopy, MmaAtom, TiledMma, auto_copy
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
    rsqrt,
    reduce_sum,
    reduce_max,
    partition_A,
    elementwise_max,
    broadcast_to,
    fill,
    cute_atomic_add,
    make_mbarriers,
    mbarrier_wait,
    mbarrier_arrive,
)
from hidet.ir.cute import layout_auto, auto_layout
from hidet.lang.mapping import auto_map
from hidet.ir.primitives.cuda.mutex import release_seq_semaphore, acquire_seq_semaphore
from hidet.ir.primitives.cuda.atomic import atomic_add, atomic_sub
from hidet.ir.type import DataType, data_type
from hidet.utils.py import cdiv
from hidet.lang.cuda import blockIdx, syncthreads

from quant_utils import bench


def transpose(m: int, n: int, threads: int, bm: int, bn: int, dtype: Union[DataType, str] = "float32"):
    from hidet.lang import attrs

    dtype_ = data_type(dtype)
    tx = bm * bn * dtype_.nbytes

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: dtype_[m, n], b: dtype_[n, m]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn)
            attrs.cuda.dynamic_smem_bytes = 0

            bid_x = blockIdx.x
            bid_y = blockIdx.y

            # mbar = make_mbarriers(1)
            tg_a = tensor_view(a, TensorLayout((m, n), (n, 1)), "global", (bm, bn), (bid_x * bm, bid_y * bn))
            ts_a = make_tensor(dtype_, layout_auto((bm, bn)), "shared")
            tr_b = make_tensor(dtype_, layout_auto((bm, bn)), "register")
            tg_b = tensor_view(b, TensorLayout((m, n), (1, m)), "global", (bm, bn), (bid_x * bm, bid_y * bn))

            txga = partition_src(tg_a, auto_copy())
            txsa = partition_dst(ts_a, auto_copy())
            copy(auto_copy((bm, bn)), txga, txsa)  # , mbarrier=mbar[0])
            # mbarrier_arrive(mbar[0], tx)
            tAsA = partition_src(ts_a, auto_copy())
            tBrB = partition_dst(tr_b, auto_copy())
            # mbarrier_wait(mbar[0], False)
            copy(auto_copy((bm, bn)), tAsA, tBrB)
            tbrb = partition_src(tr_b, auto_copy())
            tbgb = partition_dst(tg_b, auto_copy())
            copy(auto_copy((bm, bn)), tbrb, tbgb)

    func = script_module.build()

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: dtype_[m, n], b: dtype_[n, m]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn)
            attrs.cuda.dynamic_smem_bytes = 0

            bid_x = blockIdx.x
            bid_y = blockIdx.y

            tg_a = tensor_view(a, TensorLayout((m, n), (n, 1)), "global", (bm, bn), (bid_x * bm, bid_y * bn))
            tr_a = make_tensor(dtype_, layout_auto((bm, bn)), "register")
            tg_b = tensor_view(b, TensorLayout((m, n), (1, m)), "global", (bm, bn), (bid_x * bm, bid_y * bn))

            txga = partition_src(tg_a, auto_copy())
            txra = partition_dst(tr_a, auto_copy())
            copy(auto_copy((bm, bn)), txga, txra)
            tr_b = rearrange(txra, auto_layout, "register")
            tbrb = partition_src(tr_b, auto_copy())
            tbgb = partition_dst(tg_b, auto_copy())
            copy(auto_copy((bm, bn)), tbrb, tbgb)

    func = script_module.build()
    return func

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: dtype_[m, n], b: dtype_[n, m]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn)
            attrs.cuda.dynamic_smem_bytes = 0

            bid_x = blockIdx.x
            bid_y = blockIdx.y

            tg_a = tensor_view(a, TensorLayout((m, n), (n, 1)), "global", (bm, bn), (bid_x * bm, bid_y * bn))
            tr_a = make_tensor(dtype_, layout_auto((bm, bn)), "register")
            tr_b = make_tensor(dtype_, layout_auto((bm, bn)), "register")
            ts_a = make_tensor(dtype_, layout_auto((bm, bn)), "shared")
            tg_b = tensor_view(b, TensorLayout((m, n), (1, m)), "global", (bm, bn), (bid_x * bm, bid_y * bn))

            txga = partition_src(tg_a, auto_copy())
            txra = partition_dst(tr_a, auto_copy())
            tArA = partition_src(tr_a, auto_copy())
            copy(auto_copy((bm, bn)), txga, txra)
            txsa = partition_dst(ts_a, auto_copy())
            copy(auto_copy((bm, bn)), tArA, txsa)
            syncthreads()
            tAsA = partition_src(ts_a, auto_copy())
            tBrB = partition_dst(tr_b, auto_copy())
            copy(auto_copy((bm, bn)), tAsA, tBrB)
            tbrb = partition_src(tr_b, auto_copy())
            tbgb = partition_dst(tg_b, auto_copy())
            copy(auto_copy((bm, bn)), tbrb, tbgb)

    func = script_module.build()
    return func


def fused_add_rmsnorm(batch_size, seqlen, hidden_size, hd_parallel_parts):
    from hidet.lang.types import u32, i32, f16, f32
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    from hidet.lang import grid

    bm = 1
    bhd = 1024
    threads = 128

    hd_partition = bhd
    while hd_partition * hd_parallel_parts < hidden_size:
        hd_partition += bhd
    num_tokens = batch_size * seqlen
    tm = cdiv(num_tokens, bm)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            x: f16[batch_size * seqlen, hidden_size],
            residual: f16[batch_size * seqlen, hidden_size],
            weight: f16[hidden_size],
            sum_part: f32[hd_parallel_parts, batch_size * seqlen],
            sum_: f32[batch_size * seqlen],
            lock: i32[tm],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = hd_parallel_parts, tm
            attrs.cuda.dynamic_smem_bytes = 0

            hd_part = blockIdx.x
            pid_tk = blockIdx.y
            if hd_part == 0 and threadIdx.x == 0:
                lock[pid_tk] = 0

            tg_x = tensor_view(
                x[pid_tk * bm :, hd_part * hd_partition :], TensorLayout((bm, hd_partition), (hidden_size, 1)), "global"
            )
            tg_res = tensor_view(
                residual[pid_tk * bm :, hd_part * hd_partition :],
                TensorLayout((bm, hd_partition), (hidden_size, 1)),
                "global",
            )
            txgx = partition_src(tg_x, auto_copy())
            txgres = partition_src(tg_res, auto_copy())
            tr_sum = make_tensor("float32", layout_auto((bm, bhd)), "register")
            fill(tr_sum, 0.0)

            ho = (hd_partition + bhd - 1) // bhd
            for j in range(ho):
                tr_x = make_tensor("float16", layout_auto((bm, bhd)), "register")
                tr_res = make_tensor("float16", layout_auto((bm, bhd)), "register")
                txrx = partition_dst(tr_x, auto_copy())
                txrres = partition_dst(tr_res, auto_copy())
                copy(auto_copy((bm, bhd)), txgx[:, :, j], txrx)
                copy(auto_copy((bm, bhd)), txgres[:, :, j], txrres)

                imm = cast(tr_x, f32) + cast(tr_res, f32)
                x_res = cast(imm, f16)
                tr_sum = tr_sum + (imm * imm)
                txrx_res = partition_src(x_res, auto_copy())
                copy(auto_copy((bm, bhd)), txrx_res, txgres[:, :, j])

            tr_sum_total = reduce_sum(tr_sum, 1)

            counter = ~lock[pid_tk]
            if hd_part >= 1:
                tg_sum_part = tensor_view(
                    sum_part[hd_part - 1, pid_tk * bm :], TensorLayout((bm, bhd), (1, 0)), "global"
                )
                txgsum = partition_dst(tg_sum_part, auto_copy())
                txrsum = partition_src(tr_sum_total, auto_copy())
                copy(auto_copy((bm, bhd)), txrsum, txgsum)
                if threadIdx.x == 0:
                    atomic_add(counter, 1)
            else:
                acquire_seq_semaphore(counter, hd_parallel_parts - 1)
                for p in range(hd_parallel_parts - 1):
                    tg_sum_part = tensor_view(sum_part[p, pid_tk * bm :], TensorLayout((bm, bhd), (1, 0)), "global")
                    tr_sum_part = make_tensor("float32", layout_auto((bm, bhd), (1, 0)), "register")
                    txgsum = partition_src(tg_sum_part, auto_copy())
                    txrsum = partition_dst(tr_sum_part, auto_copy())
                    copy(auto_copy((bm, bhd)), txgsum, txrsum)
                    tr_sum_total = tr_sum_part + tr_sum_total
                tg_sum = tensor_view(sum_[pid_tk * bm :], TensorLayout((bm, bhd), (1, 0)), "global")
                txgsum = partition_dst(tg_sum, auto_copy())
                txrsum = partition_src(tr_sum_total, auto_copy())
                copy(auto_copy((bm, bhd)), txrsum, txgsum)
                release_seq_semaphore(counter, hd_parallel_parts)

            if hd_part >= 1:
                acquire_seq_semaphore(counter, hd_parallel_parts)
            tg_sum = tensor_view(sum_[pid_tk * bm :], TensorLayout((bm, bhd), (1, 0)), "global")
            txgsum = partition_dst(tg_sum, auto_copy())
            txrsum = partition_dst(tr_sum_total, auto_copy())
            copy(auto_copy((bm, bhd)), txgsum, txrsum)

            tg_w = tensor_view(weight[hd_part * hd_partition :], TensorLayout((bm, hd_partition), (0, 1)), "global")
            txgw = partition_src(tg_w, auto_copy())
            for j in range(ho):
                tr_x = make_tensor("float16", layout_auto((bm, bhd)), "register")
                txrx = partition_dst(tr_x, auto_copy())
                copy(auto_copy((bm, bhd)), txgres[:, :, j], txrx)
                tr_w = make_tensor("float16", layout_auto((bm, bhd), (0, 1)), "register")
                txrw = partition_dst(tr_w, auto_copy())
                copy(auto_copy((bm, bhd)), txgw[:, :, j], txrw)
                tr_norm = (cast(tr_x, f32) * cast(tr_w, f32)) * rsqrt(tr_sum_total / f32(hidden_size) + 1.0)
                tr_norm_f16 = cast(tr_norm, f16)
                txrnorm = partition_src(tr_norm_f16, auto_copy())
                copy(auto_copy((bm, bhd)), txrnorm, txgx[:, :, j])

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


def data(batch_size, seqlen, hidden_size, hd_parallel_parts, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    x = torch.randint(low=lo, high=hi, size=(batch_size * seqlen, hidden_size), dtype=dtype, device=device)
    residual = torch.randint(low=lo, high=hi, size=(batch_size * seqlen, hidden_size), dtype=dtype, device=device)
    weight = torch.randint(low=lo, high=hi, size=(hidden_size,), dtype=dtype, device=device)
    sum_ = torch.empty((batch_size * seqlen,), dtype=torch.float32, device=device)
    sum_part = torch.empty((hd_parallel_parts, batch_size * seqlen), dtype=torch.float32, device=device)

    if return_hidet:
        x = hidet.from_torch(x)
        residual = hidet.from_torch(residual)
        weight = hidet.from_torch(weight)
        sum_ = hidet.from_torch(sum_)
        sum_part = hidet.form_torch(sum_part)

    return x, residual, weight, sum_, sum_part


@pytest.mark.requires_cuda
@pytest.mark.parametrize("batch_size,seqlen,hidden_size", [(16, 1, 4096 * 4), (16, 1, 4096)])
def test_fused_add_rmsnorm(batch_size, seqlen, hidden_size):
    #    hidet.option.cache_dir("./demo_rmsnorm")
    #    hidet.option.search_space(2)
    #    hidet.option.debug_cache_tuning()
    #    hidet.option.save_lower_ir(True)
    #
    hd_parallel_parts = 2
    func = fused_add_rmsnorm(batch_size, seqlen, hidden_size, hd_parallel_parts)
    x, residual, weight, sum_, sum_part = data(batch_size, seqlen, hidden_size, hd_parallel_parts)
    residual2 = torch.empty_like(residual)
    residual2.copy_(residual)
    x2 = torch.empty_like(x)
    x2.copy_(x)
    bm = 1
    num_tokens = batch_size * seqlen
    tm = cdiv(num_tokens, bm)
    lock = torch.empty((tm,), dtype=torch.int32, device="cuda")
    mean, min_lat, max_lat = bench(func, (x, residual, weight, sum_part, sum_, lock))
    from hidet.ir.dtypes import f16

    memory = f16.nbytes * (batch_size * seqlen * hidden_size * 2 + hidden_size + batch_size * seqlen)
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    def fn():
        lock.zero_()
        func(x, residual, weight, sum_part, sum_, lock)

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    try:
        from vllm._C import ops

        def fn2():
            ops.fused_add_rms_norm(x2, residual2, weight, 1.0)

        from hidet.utils.benchmark import do_bench

        mean, min_lat, max_lat = bench(fn2, ())
        mean = do_bench(fn2, percentiles=None)
        print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))
    except ImportError:
        pass

    import numpy as np

    # verify correctness
    x, residual, weight, sum_, sum_part = data(batch_size, seqlen, hidden_size, hd_parallel_parts)
    residual2 = torch.empty_like(residual)
    residual2.copy_(residual)
    x2 = torch.empty_like(x)
    x2.copy_(x)

    def rmsnorm(x, residual):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1.0)
        x = x.to(orig_dtype) * weight
        return x, residual

    x2, residual2 = rmsnorm(x2, residual2)
    fn()
    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=residual.cpu().numpy(), desired=residual2.cpu().numpy(), rtol=1e-2)
    np.testing.assert_allclose(actual=x.cpu().numpy(), desired=x2.cpu().numpy(), rtol=1e-2)


if __name__ == "__main__":
    with hidet.option.context():
        hidet.option.cache_dir("demo_transpose")
        hidet.option.save_lower_ir(True)
        m = 32768
        n = 32768
        dtype = "float32"
        dtype_ = getattr(torch, dtype)
        op = transpose(m, n, 256, 128, 128, dtype)
        lo = 0
        hi = 3
        x = torch.randint(low=lo, high=hi, size=(m, n), dtype=dtype_, device="cuda")
        y = torch.randint(low=lo, high=hi, size=(n, m), dtype=dtype_, device="cuda")
        x = hidet.from_torch(x)
        y = hidet.from_torch(y)
        op(x, y)

        def fn():
            op(x, y)

        from hidet.utils.benchmark import do_bench

        memory = m * n * data_type(dtype).nbytes * 2
        mean = do_bench(fn, percentiles=None)
        print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

        def fn1():
            return torch.transpose(x, 1, 0).contiguous()

        memory = m * n * data_type(dtype).nbytes * 2
        mean = do_bench(fn1, percentiles=None)
        print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

        def graph(a):
            return torch.transpose(a, 1, 0).contiguous()

        options = {"triton.cudagraphs": False, "epilogue_fusion": True, "max_autotune": True}
        graph_opt = torch.compile(graph, options=options)

        def fn2():
            return graph_opt(x)

        memory = m * n * data_type(dtype).nbytes * 2
        mean = do_bench(fn2, percentiles=None)
        print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

        y2 = torch.transpose(x, 1, 0)
        print(y2)
        print(y)

        if m * n <= 1024 * 1024:
            import numpy as np

            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(actual=y.cpu().numpy(), desired=y2.cpu().numpy(), rtol=1e-2)
