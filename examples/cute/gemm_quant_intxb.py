from typing import List
import hidet

from hidet.lang import attrs
from hidet.lang.cuda import (
    blockIdx,
    threadIdx,
    syncthreads,
    cp_async_wait_all,
    cp_async_commit_group,
    cp_async_wait_group,
)
from hidet.ir.primitives.cuda.mutex import acquire_seq_semaphore, release_seq_semaphore
from hidet.ir.primitives.cuda.atomic import atomic_add
from hidet.ir.expr import var

from hidet.ir.cute.layout import TensorLayout, make_layout, layout_auto
from hidet.ir.cute.layout import Level
from hidet.ir.cute.algorithm import MmaAtom, TiledMma, auto_copy
from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    mask,
    copy,
    mma,
    rearrange,
    cast,
    fill,
    elementwise_min,
    elementwise_max
)

from hidet.ir.cute import auto_layout
from hidet.ir.cute import composition, coalesce

from hidet.utils.py import cdiv
from hidet.utils import initialize

from hidet.lang.types import i32, f32, f16, u1, u2, u4, i1, i2, i4, i8
from hidet.ir.type import DataType

from hidet.ir.library import tune

from quant_utils import (
    gemm_quant_module,
    weight_quantization_subbyte,
    weight_dequantization_subbyte,
    canonicalize,
)


_predefined_tiled_mma: List[TiledMma] = []


@initialize()
def register_tiled_mma():
    a = TensorLayout(((4, 8), (4, 2)), ((32, 1), (8, 128)))
    b = TensorLayout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 32), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 2), TensorLayout((1, 2)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (4, 2)), ((32, 1), (8, 128)))
    b = TensorLayout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 32), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    b = TensorLayout(((4, 8), (4, 2)), ((32, 1), (8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 32), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    b = TensorLayout(((4, 8), (4, 2)), ((32, 1), (8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 32), a, b, c, c, (2, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (4, 2)), ((32, 1), (8, 128)))
    b = TensorLayout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 32), a, b, c, c, (2, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (4, 2)), ((32, 1), (8, 128)))
    b = TensorLayout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 32), a, b, c, c, (2, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (4, 2)), ((32, 1), (8, 128)))
    b = TensorLayout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 32), a, b, c, c, (4, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (4, 2)), ((32, 1), (8, 128)))
    b = TensorLayout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 32), a, b, c, c, (4, 2))
    warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (1, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)


class IntxbAIntxbBGemm:
    def __init__(self, m: int, k: int, n: int, A_bits: int = 8, B_bits: int = 8, scale: float = 1.0, out_dtype: DataType = f16):
        self.m = m
        self.k = k
        self.n = n
        self.A_bits = A_bits
        self.B_bits = B_bits
        self.scale = scale
        self.out_dtype = out_dtype

    def deduce_mem_layout(self, tiled_mma: TiledMma, block_k: int, stages: int, wdtype: DataType):
        b_shape, b_tv_layout = tiled_mma.b_tv_layout()
        b_t, b_v = canonicalize(b_tv_layout)

        from hidet.transforms.cute.cuda.instruction_selection import memory_instructions
        from hidet.ir.cute.layout import (
            group,
            left_inverse,
            right_inverse,
            prefix_product,
            filter,
            complement,
        )

        candidates = []
        for inst in memory_instructions:
            if inst.src_scope.is_shared() and inst.dst_scope.is_register():
                candidates.append(inst)

        cands = []
        for inst in candidates:
            dummy = var("x", ~wdtype)
            src_inst = inst.get_layout_in_element(dummy, inst.src_layout)
            dst_inst = inst.get_layout_in_element(dummy, inst.dst_layout)
            dst_thr_inst, dst_val_inst = dst_inst[0], dst_inst[1]
            if b_t.size() < dst_thr_inst.size():
                continue
            if b_v.size() < dst_val_inst.size():
                continue
            thr_inst, thr_rest = group(b_t, dst_thr_inst.size())
            val_inst, val_rest = group(b_v, dst_val_inst.size())
            cvt = coalesce(composition(make_layout(thr_inst, val_inst), left_inverse(dst_inst)))
            result_tv = composition(cvt, src_inst)
            result_thr, result_val = result_tv
            result_thr = coalesce(make_layout(result_thr, thr_rest))
            result_val = coalesce(make_layout(result_val, val_rest))
            result_tv = make_layout(result_thr, result_val)

            result_thr = filter(result_thr)
            result_val = filter(result_val)
            last_dim = result_val.size()
            shape = result_thr.shape_tuple
            stride = prefix_product(shape, last_dim)

            shape += (last_dim,)
            stride += (1,)
            mem = TensorLayout(shape, stride)
            crd2addr = coalesce(composition(mem, left_inverse(filter(result_tv))))
            m_mode, n_mode = group(crd2addr, b_shape[0])
            block = make_layout(m_mode, n_mode)
            n_shape = n_mode.shape + (block_k // n_mode.size(),)
            n_stride = n_mode.stride + (block.cosize(),)
            n_mode_ = TensorLayout(n_shape, n_stride)
            smem_layout = make_layout(m_mode, n_mode_)
            if stages > 1:
                stage_layout = TensorLayout(stages, smem_layout.cosize())
                smem_layout = make_layout(m_mode, n_mode_, stage_layout)

            n_shape = n_mode.shape + (self.k // n_mode.size(),)
            n_stride = n_mode.stride + (block.cosize(),)
            n_mode_ = TensorLayout(n_shape, n_stride)
            gmem_layout = make_layout(m_mode, n_mode_)

            cands.append((inst, smem_layout, gmem_layout))

        inst, smem_layout, gmem_layout = cands[-1]
        return smem_layout, gmem_layout

    def get_gmem(self, gmem_layout):
        m_mode, n_mode = gmem_layout
        m_shape = m_mode.shape_tuple + (self.n // m_mode.size(),)
        m_stride = m_mode.stride_tuple + (gmem_layout.cosize(),)
        m_mode = TensorLayout(m_shape, m_stride)
        return make_layout(n_mode, m_mode)

    def modules(self):
        return tune.extract_ir_modules(self._candidates)

    @tune.space(
        2,
        tiled_mma=_predefined_tiled_mma,
        block_k=[128, 256],
        multi_buffer=[True, False],
        parallel_k_parts=[1, 2, 3, 4, 8, 16],
    )
    def _candidates(self, tiled_mma, block_k, multi_buffer, parallel_k_parts):
        if multi_buffer:
            return self.multi_buffer_kernel(tiled_mma, block_k, parallel_k_parts)
        else:
            return self.single_buffer_kernel(tiled_mma, block_k, parallel_k_parts)

    def _k_partition(self, tiled_mma: TiledMma, block_k: int, parallel_k_parts: int):
        k = self.k
        if parallel_k_parts == 1:
            return k

        k_partition = block_k
        while k_partition * parallel_k_parts < k:
            k_partition += block_k
        return k_partition

    def _get_dtype(self, bits: int):
        if bits == 1:
            return i1
        elif bits == 2:
            return i2
        elif bits == 4:
            return i4
        elif bits == 8:
            return i8

    def single_buffer_kernel(self, tiled_mma: TiledMma, block_k: int, parallel_k_parts: int):
        m, n, k = self.m, self.n, self.k
        a_shape, a_tv_layout = tiled_mma.a_tv_layout()
        b_shape, b_tv_layout = tiled_mma.b_tv_layout()
        c_shape, c_tv_layout = tiled_mma.c_tv_layout()

        block_m, inst_k = a_shape
        block_n, inst_k_ = b_shape
        block_m_, block_n_ = c_shape
        assert block_m == block_m_ and block_n == block_n_ and inst_k == inst_k_

        a_t, a_v = canonicalize(a_tv_layout)
        b_t, b_v = canonicalize(b_tv_layout)
        c_t, c_v = canonicalize(c_tv_layout)

        threads = c_t.size()

        tune.check((m == 1 and block_m == 8) or (m == block_m))
        a_dtype = self._get_dtype(self.A_bits)
        b_dtype = self._get_dtype(self.B_bits)
        k_partition = self._k_partition(tiled_mma, block_k, parallel_k_parts)

        acc_dtype = i32
        epilog_dtype = f32
        out_dtype = self.out_dtype
        need_cast_b = a_dtype != b_dtype

        qmod = None
        dqmod = None
        if self.B_bits < 8:
            smem_layout_s4, gmem_layout_s4 = self.deduce_mem_layout(tiled_mma, block_k, 1, b_dtype)
            gmem_s4 = self.get_gmem(gmem_layout_s4)
            qmod = weight_quantization_subbyte(k, n, gmem_s4, b_dtype)
            dqmod = weight_dequantization_subbyte(k, n, gmem_s4, b_dtype)
        else:
            gmem_layout_s4 = TensorLayout((block_n, k), (k, 1))
            smem_layout_s4 = layout_auto((block_n, block_k))

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: a_dtype[m, k],
                b: b_dtype[n, k],
                c: out_dtype[m, n],
                c_parallel_k_parts: epilog_dtype[parallel_k_parts, m, n],
                lock: i32[cdiv(m, block_m), cdiv(n, block_n)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(m, block_m) * cdiv(n, block_n), parallel_k_parts
                attrs.cuda.dynamic_smem_bytes = 0

                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(m, block_m)
                num_pid_n = cdiv(n, block_n)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_part = blockIdx.y
                if k_part == 0 and threadIdx.x == 0:
                    lock[pid_m, pid_n] = 0
                k_start_pos = k_part * k_partition
                k_start_ofs = gmem_layout_s4((0, k_start_pos))

                tr_a = make_tensor(a_dtype, layout_auto((block_m, inst_k * 2)), "register")
                tr_b = make_tensor(b_dtype, layout_auto((block_n, inst_k * 2)), "register")
                tr_c = make_tensor(acc_dtype, auto_layout, "register")
                fill(tr_c, 0)

                ts_a = make_tensor(a_dtype, layout_auto((block_m, block_k)), "shared")
                ts_b = make_tensor(b_dtype, smem_layout_s4, "shared")

                tg_a = tensor_view(
                    a[pid_m * block_m :, k_start_pos:], TensorLayout((block_m, k), (k, 1)), "global"
                )
                tg_b = tensor_view(
                    b[pid_n * block_n :, k_start_ofs:], gmem_layout_s4, "global"
                )

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                msk_a = mask(auto_copy(), [m - pid_m * block_m, i32(block_k)])
                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + block_k - 1) // block_k
                k_tile_max = block_k // inst_k
                for ko in range(k_block_max):
                    copy(auto_copy((block_m, block_k)), txga[:, :, ko], txsa, msk_a)
                    copy(auto_copy((block_n, block_k)), txgb[:, :, ko], txsb)
                    cp_async_wait_all()
                    syncthreads()

                    copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])
                    copy(auto_copy(), txSb[:, :, 0], txrb[:, :, 0])

                    for ki in range(k_tile_max):
                        if ki < k_tile_max - 1:
                            copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSb[:, :, ki + 1], txrb[:, :, (ki + 1) % 2])

                        if need_cast_b:
                            txrb_cvt = cast(txrb[:, :, ki % 2], a_dtype)
                            mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_cvt, tr_c)
                        else:
                            mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)
                    syncthreads()

                tr_C = cast(rearrange(tr_c, auto_layout, "register"), epilog_dtype)
                msk_c = mask(auto_copy(), [m - pid_m * block_m, n - pid_n * block_n])

                k_part = blockIdx.y
                lc = ~lock[pid_m, pid_n]
                if k_part < parallel_k_parts - 1:

                    tg_c = tensor_view(
                        c_parallel_k_parts[
                            k_part,
                            pid_m * block_m : (pid_m + 1) * block_m,
                            pid_n * block_n : (pid_n + 1) * block_n,
                        ],
                        TensorLayout((block_m, block_n), (n, 1)),
                        "global",
                    )

                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((block_m, block_n)), txrx_c, txgx_c, msk_c)

                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lc, 1)
                else:
                    tr_c_k_part = make_tensor(epilog_dtype, auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[
                                i,
                                pid_m * block_m : (pid_m + 1) * block_m,
                                pid_n * block_n : (pid_n + 1) * block_n,
                            ],
                            TensorLayout((block_m, block_n), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((block_m, block_n)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * block_m : (pid_m + 1) * block_m, pid_n * block_n : (pid_n + 1) * block_n],
                        TensorLayout((block_m, block_n), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    tr_C_clip = elementwise_max(elementwise_min(tr_C * self.scale, a_dtype.max_value), a_dtype.min_value)
                    txrx_c_final = partition_src(cast(tr_C_clip, out_dtype), auto_copy())
                    copy(auto_copy((block_m, block_n)), txrx_c_final, txgx_c_final, msk_c)

        return gemm_quant_module(script_module.ir_module(), qmod, dqmod)

    def multi_buffer_kernel(self, tiled_mma: TiledMma, block_k: int, parallel_k_parts: int):
        m, n, k = self.m, self.n, self.k
        a_shape, a_tv_layout = tiled_mma.a_tv_layout()
        b_shape, b_tv_layout = tiled_mma.b_tv_layout()
        c_shape, c_tv_layout = tiled_mma.c_tv_layout()

        block_m, inst_k = a_shape
        block_n, inst_k_ = b_shape
        block_m_, block_n_ = c_shape
        assert block_m == block_m_ and block_n == block_n_ and inst_k == inst_k_

        a_t, a_v = canonicalize(a_tv_layout)
        b_t, b_v = canonicalize(b_tv_layout)
        c_t, c_v = canonicalize(c_tv_layout)

        threads = c_t.size()

        a_dtype = self._get_dtype(self.A_bits)
        b_dtype = self._get_dtype(self.B_bits)
        dynamic_smem_bytes = (block_m * block_k * a_dtype.nbits + block_n * block_k * b_dtype.nbits) // 8
        compute_capability = hidet.option.cuda.get_arch_pair()
        compute_capability = compute_capability[0] * 10 + compute_capability[1]
        smem_limits = {
            70: 96000,
            72: 96000,
            75: 64000,
            80: 163000,
            86: 99000,
            87: 163000,
            89: 99000,
            90: 227000,
        }
        max_smem = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        stages = max_smem // dynamic_smem_bytes
        tune.check(2 <= stages < 10)
        tune.check((m == 1 and block_m == 8) or (m == block_m))
        dynamic_smem_bytes *= stages

        acc_dtype = i32
        epilog_dtype = f32
        out_dtype = self.out_dtype
        need_cast_b = a_dtype != b_dtype

        qmod = None
        dqmod = None
        if self.B_bits < 8:
            smem_layout_s4, gmem_layout_s4 = self.deduce_mem_layout(tiled_mma, block_k, stages, b_dtype)
            gmem_s4 = self.get_gmem(gmem_layout_s4)
            qmod = weight_quantization_subbyte(k, n, gmem_s4, b_dtype)
            dqmod = weight_dequantization_subbyte(k, n, gmem_s4, b_dtype)
        else:
            gmem_layout_s4 = TensorLayout((block_n, k), (k, 1))
            smem_layout_s4 = layout_auto((block_n, block_k, stages))

        k_partition = self._k_partition(tiled_mma, block_k, parallel_k_parts)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: a_dtype[m, k],
                b: b_dtype[n, k],
                c: out_dtype[m, n],
                c_parallel_k_parts: epilog_dtype[parallel_k_parts, m, n],
                lock: i32[cdiv(m, block_m), cdiv(n, block_n)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(m, block_m) * cdiv(n, block_n), parallel_k_parts
                attrs.cuda.dynamic_smem_bytes = 0

                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(m, block_m)
                num_pid_n = cdiv(n, block_n)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_part = blockIdx.y
                if k_part == 0 and threadIdx.x == 0:
                    lock[pid_m, pid_n] = 0
                k_start_pos = k_part * k_partition
                k_start_ofs = gmem_layout_s4((0, k_start_pos))

                tr_a = make_tensor(a_dtype, layout_auto((block_m, inst_k * 2)), "register")
                tr_b = make_tensor(b_dtype, layout_auto((block_n, inst_k * 2)), "register")
                tr_c = make_tensor(acc_dtype, auto_layout, "register")
                fill(tr_c, 0)

                ts_a = make_tensor(a_dtype, layout_auto((block_m, block_k, stages)), "shared")
                ts_b = make_tensor(b_dtype, smem_layout_s4, "shared")

                tg_a = tensor_view(
                    a[pid_m * block_m :, k_start_pos:], TensorLayout((block_m, k), (k, 1)), "global"
                )
                tg_b = tensor_view(
                    b[pid_n * block_n :, k_start_ofs:], gmem_layout_s4, "global"
                )

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                msk_a = mask(auto_copy(), [m - pid_m * block_m, i32(block_k)])
                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + block_k - 1) // block_k
                for s in range(stages - 1):
                    if s < k_block_max:
                        copy(auto_copy((block_m, block_k)), txga[:, :, s], txsa[:, :, s], msk_a)
                        copy(auto_copy((block_n, block_k)), txgb[:, :, s], txsb[:, :, s])
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

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + block_k - 1) // block_k
                k_tile_max = block_k // inst_k
                for ko in range(k_block_max):
                    for ki in range(k_tile_max):
                        if ki == k_tile_max - 1:
                            cp_async_wait_group(allow_on_fly_groups=stages - 2)
                            syncthreads()

                        k_tile_next = (ki + 1) % k_tile_max
                        copy(auto_copy(), txSa[:, :, k_tile_next, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSb[:, :, k_tile_next, smem_pipe_read], txrb[:, :, (ki + 1) % 2])
                        if ki == 0:
                            if ko + stages - 1 < k_block_max:
                                copy(
                                    auto_copy((block_m, block_k)),
                                    txga[:, :, ko + stages - 1],
                                    txsa[:, :, smem_pipe_write],
                                    msk_a,
                                )
                                copy(
                                    auto_copy((block_n, block_k)),
                                    txgb[:, :, ko + stages - 1],
                                    txsb[:, :, smem_pipe_write],
                                )
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        if ki == k_tile_max - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                        if need_cast_b:
                            txrb_cvt = cast(txrb[:, :, ki % 2], a_dtype)
                            mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_cvt, tr_c)
                        else:
                            mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)

                tr_C = cast(rearrange(tr_c, auto_layout, "register"), epilog_dtype)
                msk_c = mask(auto_copy(), [m - pid_m * block_m, n - pid_n * block_n])

                k_part = blockIdx.y
                lc = ~lock[pid_m, pid_n]
                if k_part < parallel_k_parts - 1:

                    tg_c = tensor_view(
                        c_parallel_k_parts[
                            k_part,
                            pid_m * block_m : (pid_m + 1) * block_m,
                            pid_n * block_n : (pid_n + 1) * block_n,
                        ],
                        TensorLayout((block_m, block_n), (n, 1)),
                        "global",
                    )

                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((block_m, block_n)), txrx_c, txgx_c, msk_c)

                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lc, 1)
                else:
                    tr_c_k_part = make_tensor(epilog_dtype, auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[
                                i,
                                pid_m * block_m : (pid_m + 1) * block_m,
                                pid_n * block_n : (pid_n + 1) * block_n,
                            ],
                            TensorLayout((block_m, block_n), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((block_m, block_n)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * block_m : (pid_m + 1) * block_m, pid_n * block_n : (pid_n + 1) * block_n],
                        TensorLayout((block_m, block_n), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    tr_C_clip = elementwise_max(elementwise_min(tr_C * self.scale, a_dtype.max_value), a_dtype.min_value)
                    txrx_c_final = partition_src(cast(tr_C_clip, out_dtype), auto_copy())
                    copy(auto_copy((block_m, block_n)), txrx_c_final, txgx_c_final, msk_c)

        return gemm_quant_module(script_module.ir_module(), qmod, dqmod)
