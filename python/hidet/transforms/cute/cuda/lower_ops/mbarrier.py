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
from typing import List, Union

from hidet.ir.expr import Expr, is_constant, var, logical_and, logical_or
from hidet.ir.dtypes import u64, u32

from hidet.ir.cute import ComposedTensorLayout, TensorLayout, product_each, right_inverse, idx2crd
from hidet.ir.cute.swizzle import Swizzle
from hidet.ir.cute.ops.copy import MBarriers, MBarrierArrive, MBarrierTryWait, MBarrierWait

from hidet.ir.primitives.cuda.barrier import (
    mbarrier_expect_transaction,
    mbarrier_arrive,
    mbarrier_try_wait,
    mbarrier_wait,
    mbarrier_init,
    fence_view_async_shared,
    fence_barrier_init,
)
from hidet.ir.primitives.cuda import cp_async_barrier_arrive
from hidet.ir.primitives.cuda import threadIdx, syncthreads, this_cluster
from hidet.ir.cute.contexts import tid_in_groups

from .registry import OpEmitter, Buffer, register_impl


WARPGROUP_SIZE = 128


@register_impl(MBarriers)
class MBarriersEmitter(OpEmitter):
    def request_smem_nbytes(self, op: MBarriers):
        return op.num_barriers * u64.nbytes

    def emit(self, op: MBarriers, args: List[Union[Buffer, Expr]], output: Buffer):
        output.buffer = self.auto_var(hint=op.name, e=self.get_smem_ptr(op, u64, 0))

        annotations = op.annotations
        if "group_ids" in annotations:
            group_ids = annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        with self.if_then(tid == 0):
            assert "num_threads" in annotations
            num_threads = annotations["num_threads"]
            if "cluster_layout" in annotations:
                cluster_layout = annotations["cluster_layout"]
                cluster_shape = product_each(cluster_layout.shape_tuple)
                cluster_m, cluster_n = cluster_shape
                num_threads = (cluster_m + cluster_n - 1) * num_threads // WARPGROUP_SIZE
            with self.for_grid([op.num_barriers]) as i:
                self.append(mbarrier_init(output.buffer + i, num_threads))
            self.append(fence_view_async_shared())
            if "cluster_layout" in annotations:
                self.append(fence_barrier_init())

        if "cluster_layout" in annotations:
            self.append(this_cluster.sync())
        else:
            self.append(syncthreads())


@register_impl(MBarrierArrive)
class MBarrierArriveEmitter(OpEmitter):
    def emit(self, op: MBarrierArrive, args: List[Union[Buffer, Expr]], output: Buffer):
        mbarrier = args[0]
        mbarrier = mbarrier.buffer + mbarrier.offset
        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        # should have some mechanism to infer multicast
        if not is_constant(op.count) or not op.count == 0:
            with self.if_then(tid == 0):
                self.append(mbarrier_expect_transaction(mbarrier, op.count))
        annotations = op.annotations
        if 'tma_fallback_copy' in annotations:
            self.append(cp_async_barrier_arrive(mbarrier))

        if "cluster_layout" in annotations:
            cluster_layout = annotations["cluster_layout"]
            cluster_shape = product_each(cluster_layout.shape_tuple)
            cluster_size = cluster_layout.size()
            cluster_layout = right_inverse(cluster_layout)
            # We spread the mbarrier arrive to different threads in the warp group
            # thereby we can amortize the latency of mbarrier arrive
            # The thread mapping magic comes from CUTLASS
            # Please refer to the following link for more details:
            # https://github.com/NVIDIA/cutlass/blob/b244379d9b15574e07b73b814b88bd2233f0b3ce/include/cutlass/pipeline/sm90_pipeline.hpp#L100-L110
            max_cluster_size = 16
            num_signaling_threads = WARPGROUP_SIZE // max_cluster_size
            layout = ComposedTensorLayout(TensorLayout((4, 4), (4, 1)), 0, Swizzle(2, 0, -2))
            tid_in_warpgroup = tid % WARPGROUP_SIZE
            signaling_index = var("signaling_index", u32)
            self.declare(signaling_index, tid_in_warpgroup // num_signaling_threads)
            thread_row = signaling_index // 4
            thread_col = signaling_index % 4
            dst_blockid = var("dst_blockid", u32)
            self.declare(dst_blockid, layout(thread_row, thread_col))
            cluster_id = var("cluster_id", u32)
            self.declare(cluster_id, this_cluster.block_rank)
            conds = []
            conds.append(dst_blockid < cluster_size)
            conds.append(tid_in_warpgroup % num_signaling_threads == 0)
            dst_m, dst_n = idx2crd(dst_blockid, cluster_shape)
            this_m, this_n = idx2crd(cluster_id, cluster_shape)
            conds.append(logical_or(dst_m == this_m, dst_n == this_n))
            self.append(mbarrier_arrive(mbarrier, dst_blockid, logical_and(*conds)))
        else:
            self.append(mbarrier_arrive(mbarrier))


@register_impl(MBarrierTryWait)
class MBarrierTryWaitEmitter(OpEmitter):
    def emit(self, op: MBarrierTryWait, args: List[Union[Buffer, Expr]], output: Buffer):
        mbarrier = args[0]
        mbarrier = mbarrier.buffer + mbarrier.offset
        phase = u32(op.phase)
        self.buffer_store(output.buffer, [0], mbarrier_try_wait(mbarrier, phase))


@register_impl(MBarrierWait)
class MBarrierWaitEmitter(OpEmitter):
    def emit(self, op: MBarrierWait, args: List[Union[Buffer, Expr]], output: Buffer):
        mbarrier = args[0]
        mbarrier = mbarrier.buffer + mbarrier.offset
        phase = u32(op.phase)
        self.append(mbarrier_wait(mbarrier, phase))
