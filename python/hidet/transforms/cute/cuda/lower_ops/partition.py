# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple, List, Union, Dict
from hidet.ir.expr import Expr, if_then_else
from hidet.ir.type import PointerType, TensorType
from hidet.ir.tools import infer_type, simplify
from hidet.lang.cuda import threadIdx
from hidet.ir.dtypes import u64
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

from hidet.ir.cute import compact_col_major, to_mixed_bits, idx2crd, flatten, Swizzle
from hidet.ir.cute.ops.partition import PartitionSrc, PartitionDst, PartitionA, PartitionB
from hidet.ir.cute import (
    coalesce,
    composition,
    canonicalize_thread_value_layout,
    TensorLayout,
    ComposedTensorLayout,
    LayoutBase,
)
from hidet.ir.cute.contexts import tid_in_groups

from hidet.utils import initialize
from .registry import OpEmitter, Buffer, register_impl


def make_swizzle_strides(flag, Z, Y, offset, I):
    if flag:
        return tuple(if_then_else(offset & (Y << i) == 0, Z * (1 << i), -Z * (1 << i)) for i in range(I))
    else:
        return tuple(if_then_else(offset & (Z << i) == 0, (Y + Z) * (1 << i), (Y - Z) * (1 << i)) for i in range(I))


def partition(
    emitter: OpEmitter,
    layout: LayoutBase,
    thread_layout: TensorLayout,
    value_layout: TensorLayout,
    tid: Expr,
    offset: Expr | None = None,
) -> Tuple[LayoutBase, Expr]:
    if isinstance(layout, ComposedTensorLayout) and isinstance(layout.functor, Swizzle):
        diced_layout = composition(layout, thread_layout)
        sliced_layout = value_layout
        swizzle = layout.functor
        B = swizzle.bits
        M = swizzle.base
        S = swizzle.shift
        shapes = (1 << M, 1 << B, 1 << (abs(S) - B), 1 << B, 1)
        strides = compact_col_major(shapes)
        sw = TensorLayout(shapes, strides)
        swizzle_anti_zy = TensorLayout(shapes, (strides[0], 0, strides[2], 0, sw.size()))
        swizzle_only_zy = TensorLayout(shapes, (0, strides[1], 0, strides[3], 0))
        diced_layout_anti_zy = composition(swizzle_anti_zy, diced_layout)
        diced_layout_only_zy = composition(swizzle_only_zy, diced_layout)
        sliced_layout_only_zy = composition(swizzle_only_zy, sliced_layout)
        swizzle_active_bits = sliced_layout_only_zy(sliced_layout_only_zy.size() - 1)
        if (swizzle_active_bits & ~swizzle(swizzle_active_bits)) != 0:
            return value_layout, diced_layout(tid, base=offset)
        Z = swizzle.zzz_msk & (-swizzle.zzz_msk)
        Y = swizzle.yyy_msk & (-swizzle.yyy_msk)
        layout_offset = 0 if offset is None else offset
        diced_layout_only_zy_shape = flatten(diced_layout_only_zy.shape_tuple)
        diced_layout_only_zy_stride = flatten(diced_layout_only_zy.stride_tuple)
        mixed_bits = to_mixed_bits(
            diced_layout_only_zy_shape, diced_layout_only_zy_stride, idx2crd(tid, diced_layout_only_zy_shape)
        )
        offset_only_zy = layout_offset
        mixed = mixed_bits[0]
        for i in mixed_bits[1:]:
            mixed = mixed ^ i
        offset_only_zy = layout_offset ^ mixed
        offset_anti_zy = diced_layout_anti_zy(tid)
        strides_lo = make_swizzle_strides(Z < Y, Z, Y, offset_only_zy, B)
        strides_hi = make_swizzle_strides(Z > Y, Z, Y, offset_only_zy, B)
        strides_lo = tuple(simplify(s) for s in strides_lo)
        strides_hi = tuple(simplify(s) for s in strides_hi)
        strides_lo_vars = []
        for s in strides_lo:
            v = emitter.auto_var(hint=f"strides_lo", e=s)
            strides_lo_vars.append(v)
        strides_hi_vars = []
        for s in strides_hi:
            v = emitter.auto_var(hint=f"strides_hi", e=s)
            strides_hi_vars.append(v)
        strides_lo = tuple(strides_lo_vars)
        strides_hi = tuple(strides_hi_vars)
        swizzle_shape = (
            (1 << M,) + tuple(2 for _ in range(B)) + (1 << (abs(S) - B),) + tuple(2 for _ in range(B)) + (1,)
        )
        swizzle_strides = (1,) + strides_lo + (1 << (M + B),) + strides_hi + (1 << (M + B + abs(S)),)
        swizzle_layout = TensorLayout(swizzle_shape, swizzle_strides)
        new_value_layout = composition(swizzle_layout, sliced_layout)
        new_offset = swizzle(offset_only_zy) + offset_anti_zy
        return new_value_layout, new_offset
    else:
        composed_layout = composition(layout, thread_layout)
        return value_layout, composed_layout(tid, base=offset)


@register_impl(PartitionSrc)
class PartitionSrcEmitter(OpEmitter):
    def emit(self, op: PartitionSrc, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_buf = src.buffer
        src_off = src.offset
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, (PointerType, TensorType))
        if isinstance(src_ty, TensorType):
            indices = [0] * len(src_ty.shape)
            src_buf = ~src_buf[indices]
        _, src_thrval_layout = op.tiled_copy.src_tv_layout()
        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        if src.scope.is_register():
            dst.buffer = src_buf
            assert dst.offset is None
        else:
            thread_layout, _ = canonicalize_thread_value_layout(src_thrval_layout)
            new_dst_layout, offset = partition(self, src.layout, thread_layout, dst.layout, tid, src_off)
            dst.buffer = src_buf
            dst.offset = self.auto_var(hint=op.name, e=offset)
            if new_dst_layout is not dst.layout:
                dst.layout = new_dst_layout
            # thr_layout = composition(src.layout, src_thrval_layout[0][0])
            # print(f"src.layout: {src.layout}")
            # if isinstance(src.layout, ComposedTensorLayout):
            #    diced_layout = composition(src.layout, src_thrval_layout[0][0])
            #    sliced_layout = dst.layout
            #    print(f"dst.layout: {dst.layout}")
            #    swizzle = src.layout.functor
            #    B = swizzle.bits
            #    M = swizzle.base
            #    S = swizzle.shift
            #    shapes = (1<<M, 1<<B, 1<<(abs(S) - B), 1<<B, 1)
            #    strides = compact_col_major(shapes)
            #    sw = TensorLayout(shapes, strides)
            #    swizzle_anti_zy = TensorLayout(shapes, (strides[0], 0, strides[2], 0, sw.size()))
            #    swizzle_only_zy = TensorLayout(shapes, (0, strides[1], 0, strides[3], 0))
            #    diced_layout_anti_zy = composition(swizzle_anti_zy, diced_layout)
            #    diced_layout_only_zy = composition(swizzle_only_zy, diced_layout)
            #    sliced_layout_only_zy = composition(swizzle_only_zy, sliced_layout)
            #    swizzle_active_bits = sliced_layout_only_zy(sliced_layout_only_zy.size() - 1)
            #    print(f"diced_layout_anti_zy: {diced_layout_anti_zy}")
            #    print(f"diced_layout_only_zy: {diced_layout_only_zy}")
            #    print(f"sliced_layout_only_zy: {sliced_layout_only_zy}")
            #    print(f"swizzle_active_bits: {swizzle_active_bits}")
            #    print(f"swizzle: {swizzle}")
            #    print(f"swizzle_active_bits & ~swizzle(swizzle_active_bits): {swizzle_active_bits & ~swizzle(swizzle_active_bits)}")
            #    print(f"swizzle_active_bits & ~swizzle(swizzle_active_bits) == 0: {swizzle_active_bits & ~swizzle(swizzle_active_bits) == 0}")
            #    print(f"swizzle_active_bits & ~swizzle(swizzle_active_bits) == 0: {swizzle_active_bits & ~swizzle(swizzle_active_bits) == 0}")
            #    Z = swizzle.zzz_msk & (-swizzle.zzz_msk)
            #    Y = swizzle.yyy_msk & (-swizzle.yyy_msk)
            #    layout_offset = 0 if src_off is None else src_off
            #    diced_layout_only_zy_shape = flatten(diced_layout_only_zy.shape_tuple)
            #    diced_layout_only_zy_stride = flatten(diced_layout_only_zy.stride_tuple)
            #    mixed_bits = to_mixed_bits(diced_layout_only_zy_shape, diced_layout_only_zy_stride, idx2crd(tid, diced_layout_only_zy_shape))
            #    offset_only_zy = layout_offset
            #    mixed = mixed_bits[0]
            #    for i in mixed_bits[1:]:
            #        mixed = mixed ^ i
            #    offset_only_zy = layout_offset ^ mixed
            #    offset_anti_zy = diced_layout_anti_zy(tid)
            #    strides_lo = make_swizzle_strides(Z < Y, Z, Y, offset_only_zy, B)
            #    strides_hi = make_swizzle_strides(Z > Y, Z, Y, offset_only_zy, B)
            #    strides_lo = tuple(simplify(s) for s in strides_lo)
            #    strides_hi = tuple(simplify(s) for s in strides_hi)
            #    strides_lo_vars = []
            #    for s in strides_lo:
            #        v = self.auto_var(hint=f"strides_lo", e=s)
            #        strides_lo_vars.append(v)
            #    strides_hi_vars = []
            #    for s in strides_hi:
            #        v = self.auto_var(hint=f"strides_hi", e=s)
            #        strides_hi_vars.append(v)
            #    strides_lo = tuple(strides_lo_vars)
            #    strides_hi = tuple(strides_hi_vars)
            #    print(f"offset_only_zy: {offset_only_zy}")
            #    print(f"offset_anti_zy: {offset_anti_zy}")
            #    print(f"strides_lo: {strides_lo}")
            #    print(f"strides_hi: {strides_hi}")
            #    swizzle_shape = (1<<M,) + tuple(2 for _ in range(B)) + (1 << (abs(S) - B), ) + tuple(2 for _ in range(B)) + (1,)
            #    swizzle_strides = (1, ) + strides_lo + (1 << (M + B), ) + strides_hi + (1 << (M + B + abs(S)), )
            #    print(f"swizzle_shape: {swizzle_shape}")
            #    print(f"swizzle_strides: {swizzle_strides}")
            #    swizzle_layout = TensorLayout(swizzle_shape, swizzle_strides)
            #    layout = composition(swizzle_layout, sliced_layout)
            #    print(f"layout: {layout}")
            #    offset = swizzle(offset_only_zy) + offset_anti_zy
            #    print(f"offset: {offset}")
            #    if (swizzle_active_bits & ~swizzle(swizzle_active_bits)) == 0:
            #        pass
            #    dst.buffer = src_buf
            #    dst.layout = layout
            #    dst.offset = self.auto_var(hint=op.name, e=offset)
            # else:
            #    dst.buffer = src_buf
            #    dst.offset = self.auto_var(hint=op.name, e=thr_layout(tid, base=src_off))

        if src.is_tma_buffer():
            assert src.scope.is_global()
            dst.tensor_maps = src.tensor_maps
            dst.coords = src.coords


@register_impl(PartitionDst)
class PartitionDstEmitter(OpEmitter):
    def emit(self, op: PartitionDst, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_buf = src.buffer
        src_off = src.offset
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, (PointerType, TensorType))
        if isinstance(src_ty, TensorType):
            indices = [0] * len(src_ty.shape)
            src_buf = ~src_buf[indices]
        _, dst_thrval_layout = op.tiled_copy.dst_tv_layout()
        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        if src.scope.is_register():
            dst.buffer = src_buf
            assert dst.offset is None
        else:
            thread_layout, _ = canonicalize_thread_value_layout(dst_thrval_layout)
            new_dst_layout, offset = partition(self, src.layout, thread_layout, dst.layout, tid, src_off)
            dst.buffer = src_buf
            dst.offset = self.auto_var(hint=op.name, e=offset)
            if new_dst_layout is not dst.layout:
                dst.layout = new_dst_layout
            # dst.buffer = src_buf
            # dst.offset = self.auto_var(hint=op.name, e=thr_layout(tid, base=src_off))

        if src.is_tma_buffer():
            assert src.scope.is_global()
            dst.tensor_maps = src.tensor_maps
            dst.coords = src.coords


# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-descriptor-format
def encode_matrix_descriptor(x):
    return (x & 0x3FFFF) >> 0x4


# build smem matrix descriptor without smem address
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-descriptor-format
def make_wgmma_desc(lead_dim_offset, stride_dim_offset, layout_type) -> int:
    desc = 0
    desc |= encode_matrix_descriptor(lead_dim_offset) << 16
    desc |= encode_matrix_descriptor(stride_dim_offset) << 32
    desc |= layout_type << 62
    return desc


wgmma_descs: Dict[str, int] = {}


@initialize()
def register_wgmma_desc_template():
    wgmma_descs["Interleaved_N"] = make_wgmma_desc(128, 256, 0)
    wgmma_descs["SW32_N"] = make_wgmma_desc(1, 256, 3)
    wgmma_descs["SW64_N"] = make_wgmma_desc(1, 512, 2)
    wgmma_descs["SW128_N"] = make_wgmma_desc(1, 1024, 1)
    wgmma_descs["Interleaved_T"] = make_wgmma_desc(128, 256, 0)
    wgmma_descs["SW32_T"] = make_wgmma_desc(512, 256, 3)
    wgmma_descs["SW64_T"] = make_wgmma_desc(1024, 512, 2)
    wgmma_descs["SW128_T"] = make_wgmma_desc(2048, 1024, 1)


@register_impl(PartitionA)
class PartitionAEmitter(OpEmitter):
    def emit(self, op: PartitionA, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_buf = src.buffer
        src_off = src.offset
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, (PointerType, TensorType))
        if isinstance(src_ty, TensorType):
            indices = [0] * len(src_ty.shape)
            src_buf = ~src_buf[indices]
        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        if src.scope.is_register():
            dst.buffer = src_buf
            assert dst.offset is None
        elif src.scope.is_shared():
            _, a_tv = op.tiled_mma.a_tv_layout()
            assert "layout_type" in op.annotations
            layout_type = op.annotations["layout_type"]
            a_t, _ = canonicalize_thread_value_layout(a_tv)
            threads = a_t.size()
            warpgroup_layout = TensorLayout((128, threads // 128), (0, 128))
            thread_layout = composition(src.layout, a_t)
            thread_layout = coalesce(composition(thread_layout, warpgroup_layout))
            smem_addr = src_buf + thread_layout(tid, base=src_off)
            smem_addr = cvta_generic_to_shared(smem_addr)
            desc_template = self.auto_var(hint='desc', e=u64(wgmma_descs[layout_type]))
            matrix_start_addr = (smem_addr & 0x3FFFF) >> 4
            matrix_base_addr = ((smem_addr >> 0x7) & 0x7) << 49
            desc = self.auto_var(hint="desc", e=desc_template | matrix_start_addr | matrix_base_addr)
            dst.buffer = desc
            dst.offset = 0


@register_impl(PartitionB)
class PartitionBEmitter(OpEmitter):
    def emit(self, op: PartitionB, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_buf = src.buffer
        src_off = src.offset
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, (PointerType, TensorType))
        if isinstance(src_ty, TensorType):
            indices = [0] * len(src_ty.shape)
            src_buf = ~src_buf[indices]
        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        if src.scope.is_register():
            dst.buffer = src_buf
            assert dst.offset is None
        elif src.scope.is_shared():
            _, b_tv = op.tiled_mma.b_tv_layout()
            assert "layout_type" in op.annotations
            layout_type = op.annotations["layout_type"]
            b_t, _ = canonicalize_thread_value_layout(b_tv)
            threads = b_t.size()
            warpgroup_layout = TensorLayout((128, threads // 128), (0, 128))
            thread_layout = composition(src.layout, b_t)
            thread_layout = coalesce(composition(thread_layout, warpgroup_layout))
            smem_addr = src_buf + thread_layout(tid, base=src_off)
            smem_addr = cvta_generic_to_shared(smem_addr)
            desc_template = self.auto_var(hint='desc', e=u64(wgmma_descs[layout_type]))
            matrix_start_addr = (smem_addr & 0x3FFFF) >> 4
            matrix_base_addr = ((smem_addr >> 0x7) & 0x7) << 49
            desc = self.auto_var(hint="desc", e=desc_template | matrix_start_addr | matrix_base_addr)
            dst.buffer = desc
            dst.offset = 0
