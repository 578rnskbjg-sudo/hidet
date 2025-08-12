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

"""
This module provides functionality for planning cluster layouts in CUDA code.


"""
from typing import Dict, Set, List

from hidet.ir.expr import Var
from hidet.ir.stmt import DeclareStmt
from hidet.ir.functors import IRVisitor, IRRewriter
from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass

from hidet.ir.cute.expr import Op, CallOp
from hidet.ir.cute.ops import Mma, Copy, Partition, SubTensor, Arithmetic, MBarriers, MBarrierArrive
from hidet.ir.cute import TensorLayout, right_inverse, product_each, flatten, shape_div
from hidet.transforms.cute.analysis import TensorInfo, TensorAliasAnalysis


class ClusterInfoPlanner(IRVisitor):
    def __init__(self, var2tensor: Dict[Var, TensorInfo]):
        super().__init__()
        self.var2tensor: Dict[Var, TensorInfo] = var2tensor

        self.op2cluster_layout: Dict[Op, TensorLayout] = {}

        self.output_var2op: Dict[Var, Op] = {}
        self.adjacent_ops: Dict[Op, Set[Op]] = {}
        self.full_barriers: Set[MBarriers] = set()
        self.mbarriers: List[MBarriers] = []
        self.arrival2mbar: Dict[MBarrierArrive, MBarriers] = {}
        self.consumer_cluster_layout: TensorLayout = None

    def visit_Mma(self, op: Mma):
        if op.cluster_layout.size() > 1:
            self.consumer_cluster_layout = op.cluster_layout
            cluster_layout = op.cluster_layout
            op_a = self.output_var2op[op.a]
            op_b = self.output_var2op[op.b]
            shape = product_each(cluster_layout.shape_tuple)
            assert len(shape) == 2
            cluster_m = shape[0]
            cluster_id2mn = right_inverse(cluster_layout)
            flat_shape = flatten(cluster_id2mn.shape_tuple)
            flat_stride = flatten(cluster_id2mn.stride_tuple)
            result_shape = []
            result_stride = []
            for s, d in zip(flat_shape, flat_stride):
                if s * d <= cluster_m or d > cluster_m:
                    result_shape.append(s)
                    result_stride.append(d)
                else:
                    s1 = shape_div(cluster_m, d)
                    result_shape.append(s1)
                    result_stride.append(d)
                    result_shape.append(shape_div(s, s1))
                    result_stride.append(d * s1)
            cluster_stride_a = [d if d < cluster_m else 0 for d in result_stride]
            cluster_stride_b = [d if d >= cluster_m else 0 for d in result_stride]
            self.op2cluster_layout[op_a] = TensorLayout(tuple(result_shape), tuple(cluster_stride_a))
            self.op2cluster_layout[op_b] = TensorLayout(tuple(result_shape), tuple(cluster_stride_b))

    def visit_Copy(self, op: Copy):
        op_src = self.output_var2op[op.src]
        op_dst = self.output_var2op[op.dst]
        self.adjacent_ops[op] = {op_src, op_dst}
        self.adjacent_ops[op_src].add(op)
        self.adjacent_ops[op_dst].add(op)

        mbarrier = op.mbarrier
        if mbarrier is not None:
            tensor_info = self.var2tensor.get(mbarrier, None)
            if tensor_info is not None:
                mbarrier = tensor_info.tensor
                assert isinstance(mbarrier, MBarriers)
                self.full_barriers.add(mbarrier)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        init = stmt.init
        if isinstance(init, CallOp):
            call = init
            op = call.op
            self.visit(op)
            self.output_var2op[stmt.var] = op
            if isinstance(op, (Partition, SubTensor, Arithmetic)):
                for arg in op.args:
                    if isinstance(arg, (list, tuple)):
                        continue
                    arg_op = self.output_var2op[arg]
                    self.adjacent_ops[op] = {arg_op}
                    if arg_op not in self.adjacent_ops:
                        self.adjacent_ops[arg_op] = {op}
                    else:
                        self.adjacent_ops[arg_op].add(op)

    def visit_MBarriers(self, op: MBarriers):
        self.mbarriers.append(op)

    def visit_MBarrierArrive(self, op: MBarrierArrive):
        mbarrier = op.mbarrier
        tensor_info = self.var2tensor.get(mbarrier, None)
        assert tensor_info is not None
        mbar_tensor = tensor_info.tensor
        assert isinstance(mbar_tensor, MBarriers)
        self.arrival2mbar[op] = mbar_tensor

    def plan(self, func: Function):
        self.visit(func)

        if len(self.op2cluster_layout.items()) == 0:
            return {}

        ready = None
        for op, _ in self.adjacent_ops.items():
            if op in self.op2cluster_layout:
                ready = op

        while ready is not None:
            cluster_layout = self.op2cluster_layout[ready]
            adjacent_ops = self.adjacent_ops[ready]
            for adjacent in adjacent_ops:
                self.op2cluster_layout[adjacent] = cluster_layout
            self.adjacent_ops.pop(ready)
            ops_to_remove = []
            for op, adjacent_ops in self.adjacent_ops.items():
                if ready in adjacent_ops:
                    adjacent_ops.remove(ready)
                if len(adjacent_ops) == 0:
                    ops_to_remove.append(op)
            for op in ops_to_remove:
                self.adjacent_ops.pop(op)

            ready = None
            for op, _ in self.adjacent_ops.items():
                if op in self.op2cluster_layout:
                    ready = op

        for mbar in self.mbarriers:
            if mbar not in self.full_barriers:
                self.op2cluster_layout[mbar] = self.consumer_cluster_layout

        for arrival, mbar in self.arrival2mbar.items():
            if mbar not in self.full_barriers:
                self.op2cluster_layout[arrival] = self.consumer_cluster_layout

        return self.op2cluster_layout


class ClusterInfoUpdater(IRRewriter):
    def __init__(self, op2cluster_layout: Dict[Op, TensorLayout]):
        super().__init__()
        self.op2cluster_layout: Dict[Op, TensorLayout] = op2cluster_layout

    def visit_Copy(self, op: Copy):
        if op in self.op2cluster_layout:
            cluster_layout = self.op2cluster_layout[op]
            src = self.visit(op.src)
            dst = self.visit(op.dst)
            if op.mask is not None:
                mask = self.visit(op.mask)
            else:
                mask = None
            if op.mbarrier is not None:
                mbarrier = self.visit(op.mbarrier)
            else:
                mbarrier = None
            annotations_update = {"cluster_layout": cluster_layout}
            return op.reforward([src, dst, mask, mbarrier], annotations_update=annotations_update)
        return super().visit_Copy(op)

    def visit_MBarriers(self, op: MBarriers):
        if op in self.op2cluster_layout:
            cluster_layout = self.op2cluster_layout[op]
            annotations_update = {"cluster_layout": cluster_layout}
            return op.reforward([], annotations_update=annotations_update)
        return super().visit_MBarriers(op)

    def visit_MBarrierArrive(self, op: MBarrierArrive):
        if op in self.op2cluster_layout:
            cluster_layout = self.op2cluster_layout[op]
            mbarrier = self.visit(op.mbarrier)
            count = self.visit(op.count)
            annotations_update = {"cluster_layout": cluster_layout}
            return op.reforward([mbarrier, count], annotations_update=annotations_update)
        return super().visit_MBarrierArrive(op)


class PlanClusterLayoutPass(FunctionPass):
    def __init__(self):
        super().__init__()

    def process_func(self, func: Function) -> Function:
        var2tensor = TensorAliasAnalysis().analyze(func)
        planner = ClusterInfoPlanner(var2tensor)
        op2cluster_layout = planner.plan(func)
        rewriter = ClusterInfoUpdater(op2cluster_layout)
        return rewriter(func)


def plan_cluster_layout_pass() -> FunctionPass:
    return PlanClusterLayoutPass()
