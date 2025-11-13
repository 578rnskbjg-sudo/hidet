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
import numpy as np

import pytest
import hidet
import torch
from hidet import ops
from hidet.ir.cute.layout import (
    TensorLayout,
    composition,
    ThrValAtom,
    Level,
    TiledTensorLayout,
    logical_divide,
    left_inverse,
)
from hidet.option import OptionContext
from hidet.ir.cute.int_tuple import compact_col_major
from hidet.utils import initialize

from quant_utils import bench


matmul_tests = []


@initialize()
def initialize_tests():
    for M in [2048, 4096]: 
        for [N, K] in [[6144, 4096], [4096, 14336], [14336, 4096], [4096, 4096], [10240, 8192], [8192, 28672], [8192, 8192], [28672, 8192]]:
            matmul_tests.append((M, N, K, 1))


def data(M, N, K, L, dtype="bfloat16", device="cuda"):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    a = torch.randint(low=lo, high=hi, size=(L, M, K), dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=(L, K, N), dtype=dtype, device=device)

    return a, b


def test_problem(M, N, K, L, dtype, cand: int | None = None):
    graph_args = data(M, N, K, L, dtype=dtype)

    from hidet.runtime.compiled_task import compiled_task_cache
    compiled_task_cache.cached = {}
    with hidet.option.context():
        hidet.option.cache_dir(f"./matmul_standalone_cand{cand if cand is not None else ''}")
        hidet.option.debug_cache_tuning()
        hidet.option.save_lower_ir(True)
        hidet.option.parallel_k(strategy='disabled')
        hidet.option.search_space(2)
        hidet.option.hexcute_candidate(cand)
        hidet.option.hexcute_matmul(strategy='enable')

        def matmul_graph():
            a = hidet.symbol([L, M ,K], dtype=dtype, device='cuda')
            b = hidet.symbol([L, K, N], dtype=dtype, device='cuda')
            c = ops.matmul(a, b)
            return hidet.trace_from(c, [a, b])

        graph_hidet = matmul_graph()
        graph_hidet = hidet.graph.optimize(graph_hidet)
        hidet_args = [hidet.from_torch(t) for t in graph_args]
        D_hidet = graph_hidet(*hidet_args)
        matmul_task = graph_hidet.get_compiled_task(0)
        D_hidet_ = matmul_task.create_outputs(hidet_args)[0]
        kernel = matmul_task.candidates[matmul_task.pick_best_candidate(hidet_args, [D_hidet_])]
        
        def hidet_matmul(a, b):
            kernel(a, b, D_hidet_)
            return D_hidet_

        hidet_mean, hidet_min, hidet_max = bench(hidet_matmul, hidet_args)
        print(f"hidet(torch.compile): {hidet_mean} ms")
        return hidet_mean


def main(cand: int | None = None):
    records = []

    records.append(cand if cand is not None else -1)
    for problem in matmul_tests:
        hidet_time = test_problem(*problem, dtype='float16', cand=cand)
        records.append(hidet_time)
    return records


if __name__ == "__main__":
    from tabulate import tabulate
    headers = ["cand"] + [f"{m}x{n}x{k}" for m, n, k, _ in matmul_tests]
    records = []
    for cand in [0, 1, 2, 3, 4, 5, None]:
        print(f"Running with cand: {cand}")
        records.append(main(cand))
        print(f"Records: {records}")
        
    with open(f"cost_model_accuracy.txt", "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )
        
    from matplotlib import pyplot as plt
    import numpy as np
    pred_cost_model = records[-1][1:]
    range_min = [None for _ in range(len(pred_cost_model))]
    range_max = [None for _ in range(len(pred_cost_model))]
    for record in records:
        for i in range(len(record[1:])):
            if range_min[i] is None:
                range_min[i] = record[i + 1]
            else:
                range_min[i] = min(range_min[i], record[i + 1])
            if range_max[i] is None:
                range_max[i] = record[i + 1]
            else:
                range_max[i] = max(range_max[i], record[i + 1])
    
    print(range_min)
    print(range_max)
    print(pred_cost_model)
    
    picked = np.array(pred_cost_model)
    best = np.array(range_min)
    worst = np.array(range_max)
    
    yerr_lower = picked - best
    yerr_upper = worst - picked
    yerr = np.vstack([yerr_lower, yerr_upper])
    
    x = np.arange(len(matmul_tests))
    layers = [f"M{i}" for i in range(len(matmul_tests))]    
    # ---------------------------------------------
    # Plot
    # ---------------------------------------------
    plt.figure(figsize=(8, 4))
    
    bars = plt.bar(x, picked, color="#74a9cf", label="Picked by cost model")
    plt.scatter(x, best, color="black", label="Best candidate (min latency)")
    plt.errorbar(x, picked, yerr=yerr, fmt="none", ecolor="#ef8a62",
                 capsize=5, label="Latency range (all candidates)")
   
    
    plt.ylabel("Latency (ms)")
    plt.xticks(x, layers)
    plt.ylim(0, worst.max() * 1.15)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("cost_model_accuracy.pdf", dpi=300, bbox_inches="tight")