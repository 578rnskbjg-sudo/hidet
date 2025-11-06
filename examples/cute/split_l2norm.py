import torch
import triton
import triton.language as tl
from typing import Optional
from quant_utils import bench
from hidet.utils.benchmark import do_bench
import hidet
from hidet.utils.py import cdiv
from hidet.lang.types import u32, i32, f16, f32
from hidet.lang import attrs, grid
from hidet.lang.cuda import blockIdx, threadIdx

from hidet.ir.cute.algorithm import auto_copy
from hidet.ir.cute import auto_layout, layout_auto
from hidet.ir.cute import TensorLayout

from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    copy,
    cast,
    reduce_sum,
    rsqrt,
    mask,
    rearrange,
)


def l2norm_fwd_kernel(num_tokens, hidden_size, num_k_heads, head_k_dim, hd_offset=0, eps=1e-6):
    MBLOCK = 32
    D = head_k_dim 
    tokens_per_block = MBLOCK // num_k_heads
   
    assert num_k_heads <= MBLOCK and MBLOCK % num_k_heads == 0
    threads = 256
    gridm = cdiv(num_tokens * num_k_heads, MBLOCK)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            x: f16[num_tokens, hidden_size],
            y: f16[num_tokens * num_k_heads, head_k_dim],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = gridm
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            token_idx = pid * tokens_per_block
            mask_x = mask(auto_copy(), [num_tokens * num_k_heads - (pid * MBLOCK), i32(D)])
            tg_x = tensor_view(
                x[token_idx:, hd_offset:], TensorLayout(((num_k_heads, MBLOCK // num_k_heads), D), ((D, hidden_size), 1)), "global"
            )
            txgx = partition_src(tg_x, auto_copy())
            tr_g = make_tensor("float16", layout_auto((MBLOCK, D)), "register")
            txrx = partition_dst(tr_g, auto_copy())
            copy(auto_copy((MBLOCK, D)), txgx, txrx, mask_x)
            
            x_f32 = cast(txrx, f32)
            x2 = x_f32 * x_f32
            x2_sum = reduce_sum(x2, axis=1)
            res = cast(x_f32 * rsqrt(x2_sum + eps), f16) 
            txrres = partition_src(res, auto_copy())
            mask_y = mask(auto_copy(), [num_tokens * num_k_heads - (pid * MBLOCK), i32(D)])
            tg_y = tensor_view(y[pid * MBLOCK:, :], TensorLayout((MBLOCK, D), (D, 1)), "global")
            txgy = partition_dst(tg_y, auto_copy())
            copy(auto_copy((MBLOCK, D)), txrres, txgy, mask_y)
    func = script_module.build()
    return func


def fused_spit_l2norm_fwd_kernel(num_tokens, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim, eps=1e-6):
    MBLOCK = 32
    D = head_k_dim
    D2 = head_v_dim 
    tokens_per_block = MBLOCK // num_k_heads
    tokens_per_block2 = MBLOCK // num_v_heads
   
    assert num_k_heads <= MBLOCK and MBLOCK % num_k_heads == 0
    threads = 256
    gridqk = cdiv(num_tokens * num_k_heads, MBLOCK)
    gridv = cdiv(num_tokens * num_v_heads, MBLOCK)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            x: f16[num_tokens, hidden_size],
            q: f16[num_tokens * num_k_heads, head_k_dim],
            k: f16[num_tokens * num_k_heads, head_k_dim],
            v: f16[num_tokens * num_v_heads, head_v_dim],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = gridqk * 2 + gridv
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            if pid < 2 * gridqk:
                hd_offset = pid // gridqk * num_k_heads * head_k_dim
                token_idx = (pid % gridqk) * tokens_per_block
                mask_x = mask(auto_copy(), [num_tokens * num_k_heads - ((pid % gridqk) * MBLOCK), i32(D)])
                tg_x = tensor_view(
                    x[token_idx:, hd_offset:], TensorLayout(((num_k_heads, MBLOCK // num_k_heads), D), ((D, hidden_size), 1)), "global"
                )
                txgx = partition_src(tg_x, auto_copy())
                tr_g = make_tensor("float16", layout_auto((MBLOCK, D)), "register")
                txrx = partition_dst(tr_g, auto_copy())
                copy(auto_copy((MBLOCK, D)), txgx, txrx, mask_x)

                x_f32 = cast(txrx, f32)
                x2 = x_f32 * x_f32
                x2_sum = reduce_sum(x2, axis=1)
                res = cast(x_f32 * rsqrt(x2_sum + eps), f16) 
                txrres = partition_src(res, auto_copy())
                if pid < gridqk:
                    mask_q = mask(auto_copy(), [num_tokens * num_k_heads - ((pid % gridqk) * MBLOCK), i32(D)])
                    tg_q = tensor_view(q[(pid % gridqk) * MBLOCK:, :], TensorLayout((MBLOCK, D), (D, 1)), "global")
                    txgq = partition_dst(tg_q, auto_copy())
                    copy(auto_copy((MBLOCK, D)), txrres, txgq, mask_q)
                else:
                    mask_k = mask(auto_copy(), [num_tokens * num_k_heads - ((pid % gridqk) * MBLOCK), i32(D)])
                    tg_k = tensor_view(k[(pid % gridqk) * MBLOCK:, :], TensorLayout((MBLOCK, D), (D, 1)), "global")
                    txgk = partition_dst(tg_k, auto_copy())
                    copy(auto_copy((MBLOCK, D)), txrres, txgk, mask_k)
            else:
                hd_offset = 2 * num_k_heads * head_k_dim
                token_idx = (pid - 2 * gridqk) * tokens_per_block2
                mask_x = mask(auto_copy(), [num_tokens * num_v_heads - ((pid - 2 * gridqk) * MBLOCK), i32(D2)])
                tg_x = tensor_view(
                    x[token_idx:, hd_offset:], TensorLayout(((num_v_heads, MBLOCK // num_v_heads), D2), ((D2, hidden_size), 1)), "global"
                )
                txgx = partition_src(tg_x, auto_copy())
                tr_g = make_tensor("float16", layout_auto((MBLOCK, D2)), "register")
                txrx = partition_dst(tr_g, auto_copy())
                copy(auto_copy((MBLOCK, D2)), txgx, txrx, mask_x)
                
                res = rearrange(txrx, auto_layout, "register")
                txrres= partition_src(res, auto_copy())
                mask_v = mask(auto_copy(), [num_tokens * num_v_heads - ((pid - 2 * gridqk) * MBLOCK), i32(D2)])
                tg_v = tensor_view(v[(pid - 2 * gridqk) * MBLOCK:, :], TensorLayout((MBLOCK, D2), (D2, 1)), "global")
                txgv = partition_dst(tg_v, auto_copy())
                copy(auto_copy((MBLOCK, D2)), txrres, txgv, mask_v)

    func = script_module.build()
    return func


@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, N)[None, :]
    xs = tl.load(X + (rindex + N * row_idx), xmask).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, N])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)


def l2norm_fwd(x: torch.Tensor,
               eps: float = 1e-6,
               output_dtype: Optional[torch.dtype] = None):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    MBLOCK = 32
    # M, N = x.shape
    l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK), )](
        x,
        y,
        eps,
        T,
        D,
        MBLOCK,
    )

    return y.view(x_shape_og)


def data(num_tokens, hidden_size, num_k_heads, head_k_dim, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    mixed_qkv = torch.randint(low=lo, high=hi, size=(num_tokens, hidden_size), dtype=dtype, device=device)
    q = torch.randint(low=lo, high=hi, size=(num_tokens, num_k_heads, head_k_dim), dtype=dtype, device=device)
    k= torch.randint(low=lo, high=hi, size=(num_tokens, num_k_heads, head_k_dim), dtype=dtype, device=device)

    if return_hidet:
        q = hidet.from_torch(q)
        k = hidet.from_torch(k)
        mixed_qkv = hidet.from_torch(mixed_qkv)

    return q, k, mixed_qkv


def main():
    with hidet.option.context():
        hidet.option.cache_dir("split_l2norm")
        hidet.option.debug_cache_tuning(True)
        hidet.option.save_lower_ir(True)
        hidet.option.search_space(2)
        hidet.option.num_local_workers(1)
        hidet.option.use_torch_stream(True)
        num_tokens = 8192
        hidden_size = 2048
        head_k_dim = 128
        head_v_dim = 128
        tp_size = 4
        key_dim = 2048 
        value_dim = 4096
        key_dim = key_dim // tp_size # 512
        value_dim = value_dim // tp_size # 1024
        num_k_heads = key_dim // head_k_dim # 4
        num_v_heads = value_dim // head_v_dim # 8

        q, k, mixed_qkv = data(num_tokens, hidden_size, num_k_heads, head_k_dim, dtype="float16", device="cuda", return_hidet=False)

        l2norm_q = l2norm_fwd_kernel(num_tokens, hidden_size, num_k_heads, head_k_dim, 0, 1e-6)
        l2norm_k = l2norm_fwd_kernel(num_tokens, hidden_size, num_k_heads, head_k_dim, key_dim, 1e-6)

        def fn_hexcute(mixed_qkv):
            v = mixed_qkv[:, 2 * key_dim:]
            q = torch.empty((1, num_tokens, num_k_heads, head_k_dim), dtype=torch.float16, device="cuda")
            k = torch.empty((1, num_tokens, num_k_heads, head_k_dim), dtype=torch.float16, device="cuda")
            l2norm_q(mixed_qkv, q)
            l2norm_k(mixed_qkv, k)
            return q, k, v.reshape(1, num_tokens, -1, head_v_dim).contiguous()

        fused_spit_l2norm_fwd = fused_spit_l2norm_fwd_kernel(num_tokens, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim, 1e-6)

        def fn_hexcute2(mixed_qkv):
            q = torch.empty((1, num_tokens, num_k_heads, head_k_dim), dtype=torch.float16, device="cuda")
            k = torch.empty((1, num_tokens, num_k_heads, head_k_dim), dtype=torch.float16, device="cuda")
            v = torch.empty((1, num_tokens, num_v_heads, head_v_dim), dtype=torch.float16, device="cuda")
            fused_spit_l2norm_fwd(mixed_qkv, q, k, v)
            return q, k, v

        from einops import rearrange
        def fn(mixed_qkv):
            q, k, v = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
            q, k = map(lambda x: rearrange(x, "l (h d) -> 1 l h d", d=head_k_dim), (q, k))
            v = rearrange(v, "l (h d) -> 1 l h d", d=head_v_dim)
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
            return q, k, v.contiguous()
 
        config = {"triton.cudagraphs": False, "epilogue_fusion": True, "max_autotune": True}
        #torch._inductor.config.combo_kernels=True
        graph_opt = torch.compile(fn, options=config)
        
        q, k, v = fn(mixed_qkv)
        q1, k1, v1 = graph_opt(mixed_qkv)
        q2, k2, v2 = fn_hexcute(mixed_qkv)
        q3, k3, v3 = fn_hexcute2(mixed_qkv)
        print(q1.shape)
        print(k1.shape)
        print(v1.shape)

        import numpy as np
        np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
        np.testing.assert_allclose(actual=q1.cpu().numpy(), desired=q.cpu().numpy(), rtol=1e-2)
        np.testing.assert_allclose(actual=k1.cpu().numpy(), desired=k.cpu().numpy(), rtol=1e-2)
        np.testing.assert_allclose(actual=v1.cpu().numpy(), desired=v.cpu().numpy(), rtol=1e-2)
        np.testing.assert_allclose(actual=q2.cpu().numpy(), desired=q.cpu().numpy(), rtol=1e-2)
        np.testing.assert_allclose(actual=k2.cpu().numpy(), desired=k.cpu().numpy(), rtol=1e-2)
        np.testing.assert_allclose(actual=v2.cpu().numpy(), desired=v.cpu().numpy(), rtol=1e-2)
        np.testing.assert_allclose(actual=q3.cpu().numpy(), desired=q.cpu().numpy(), rtol=1e-2)
        np.testing.assert_allclose(actual=k3.cpu().numpy(), desired=k.cpu().numpy(), rtol=1e-2)
        np.testing.assert_allclose(actual=v3.cpu().numpy(), desired=v.cpu().numpy(), rtol=1e-2)

        def wrapped_fn_hexcute():
            return fn_hexcute(mixed_qkv)
        mean = do_bench(wrapped_fn_hexcute, warmup=5, rep=10, percentiles=None)
        print(f"hexcute: time={mean:.3f} ms")

        def wrapped_fn_hexcute2():
            return fn_hexcute2(mixed_qkv)
        mean = do_bench(wrapped_fn_hexcute2, warmup=5, rep=10, percentiles=None)
        print(f"hexcute2: time={mean:.3f} ms")

        def wrapped_fn():
            return fn(mixed_qkv)
        mean = do_bench(wrapped_fn, warmup=5, rep=10, percentiles=None)
        print(f"split: time={mean:.3f} ms")

        def wrapped_fn2():
            return graph_opt(mixed_qkv)
        mean = do_bench(wrapped_fn2, warmup=5, rep=10, percentiles=None)
        print(f"inductor: time={mean:.3f} ms")


if __name__ == "__main__":
    main()
