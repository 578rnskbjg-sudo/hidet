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


bench_suites = "functional"


dense_matmul = [
    # inference
    (5124, 700, 2048),
    (35, 700, 2048),
    (5124, 1500, 2560),
    (35, 1500, 2560),
    (7680, 2, 2560),
    (7680, 4, 2560),
    (1024, 2, 500000),
    (512, 4, 500000),
    (1024, 700, 512),
    (7680, 1500, 2560),
    (8448, 2, 2816),
    (8448, 4, 2816),
    (3072, 1500, 128),
    (512, 1500, 2816),
    (1024, 1500, 2816),
    (512, 3000, 2816),
    (1024, 3000, 2816),
    (512, 6000, 2816),
    (1024, 6000, 2816),
    # training
    (2560, 16, 2560),
    (2560, 24, 2560),
    (2560, 128, 2560),
    (2560, 7000, 2560),
    (4096, 7000, 4096),
    (2560, 7134, 2560),
    (2048, 10376, 2048),
    (4096, 8936, 4096),
    (5124, 9124, 1760),
    (35, 8458, 1760),
    (5124, 9124, 2048),
    (35, 8458, 4096),
    (7680, 16, 2560),
    (7680, 32, 2560),
    (7680, 64, 2560),
    (7680, 128, 2560),
    (3072, 16, 1024),
    (3072, 32, 1024),
    (3072, 64, 1024),
    (3072, 128, 1024),
    (7680, 5480, 2560),
    (512, 8, 500000),
    (1024, 8, 500000),
    (512, 16, 500000),
    (1024, 16, 500000),
    (7680, 24000, 2560),
    (6144, 24000, 2048),
    (4608, 24000, 1536),
    (8448, 24000, 2816),
    (3072, 24000, 1024),
    (7680, 48000, 2560),
    (6144, 48000, 2048),
    (4608, 48000, 1536),
    (8448, 48000, 2816),
    (3072, 48000, 1024),
    (512, 24000, 2816),
    (1024, 24000, 2816),
]


# dense_matmul = [
#    # inference
#    (5124, 700, 2048),
#    (35, 700, 2048),
#    (5124, 700, 2560),
#    (35, 700, 2560),
#    (5124, 1500, 2048),
#    (35, 1500, 2048),
#    (5124, 1500, 2560),
#    (35, 1500, 2560),
#    (7680, 2, 2560),
#    (7680, 8, 2560),
#    (3072, 2, 1024),
#    (3072, 4, 1024),
#    (512, 2, 500000),
#    (1024, 2, 500000),
#    (512, 4, 500000),
#    (1024, 4, 500000),
#    (1024, 700, 512),
#    (7680, 1500, 2560),
#    (6144, 1500, 2048),
#    (4608, 1500, 1536),
#    (8448, 1500, 2816),
#    (3072, 1500, 1024),
#    (7680, 3000, 2560),
#    (6144, 3000, 2048),
#    (4608, 3000, 1536),
#    (8448, 3000, 2816),
#    (3072, 3000, 1024),
#    (7680, 6000, 2560),
#    (6144, 6000, 2048),
#    (4608, 6000, 1536),
#    (8448, 6000, 2816),
#    (3072, 6000, 1024),
#    (6144, 1, 2048),
#    (4608, 1, 1536),
#    (8448, 1, 2816),
#    (6144, 2, 2048),
#    (4608, 2, 1536),
#    (8448, 2, 2816),
#    (6144, 4, 2048),
#    (4608, 4, 1536),
#    (8448, 4, 2816),
#    (128, 1500, 1280),
#    (3072, 1500, 128),
#    (128, 1, 1024),
#    (3072, 1, 128),
#    (176, 1500, 1408),
#    (4224, 1500, 176),
#    (128, 1, 1408),
#    (4224, 1, 128),
#    (512, 1500, 2816),
#    (512, 1500, 2048),
#    (512, 1500, 2560),
#    (512, 1500, 1530),
#    (1024, 1500, 2816),
#    (1024, 1500, 2048),
#    (1024, 1500, 2560),
#    (1024, 1500, 1530),
#    (512, 1, 512),
#    (1024, 1, 512),
#    (512, 3000, 2816),
#    (512, 3000, 2048),
#    (512, 3000, 2560),
#    (512, 3000, 1530),
#    (1024, 3000, 2816),
#    (1024, 3000, 2048),
#    (1024, 3000, 2560),
#    (1024, 3000, 1530),
#    (512, 2, 512),
#    (1024, 2, 512),
#    (512, 6000, 2816),
#    (512, 6000, 2048),
#    (512, 6000, 2560),
#    (512, 6000, 1530),
#    (1024, 6000, 2816),
#    (1024, 6000, 2048),
#    (1024, 6000, 2560),
#    (1024, 6000, 1530),
#    (512, 4, 512),
#    (1024, 4, 512),
#    # training
#    (2560, 16, 2560),
#    (2560, 32, 2560),
#    (2560, 64, 2560),
#    (2560, 128, 2560),
#    (2560, 7000, 2560),
#    (2560, 16, 1760),
#    (2560, 32, 1760),
#    (2560, 64, 1760),
#    (2560, 128, 1760),
#    (2560, 7000, 1760),
#    (2560, 16, 2048),
#    (2560, 32, 2048),
#    (2560, 64, 2048),
#    (2560, 128, 2048),
#    (2560, 7000, 2048),
#    (4096, 16, 4096),
#    (4096, 32, 4096),
#    (4096, 64, 4096),
#    (4096, 128, 4096),
#    (4096, 7000, 4096),
#    (2560, 16, 2560),
#    (2560, 32, 2560),
#    (2560, 64, 2560),
#    (2560, 128, 2560),
#    (2560, 7000, 2560),
#    (1760, 16, 1760),
#    (1760, 32, 1760),
#    (1760, 64, 1760),
#    (1760, 128, 1760),
#    (1760, 7000, 1760),
#    (2048, 16, 2048),
#    (2048, 32, 2048),
#    (2048, 64, 2048),
#    (2048, 128, 2048),
#    (2048, 7000, 2048),
#    (4096, 16, 4096),
#    (4096, 32, 4096),
#    (4096, 64, 4096),
#    (4096, 128, 4096),
#    (4096, 7000, 4096),
#    (2560, 7133, 2560),
#    (1760, 6574, 1760),
#    (2048, 10376, 2048),
#    (4096, 8935, 4096),
#    (5124, 9124, 1760),
#    (35, 8457, 1760),
#    (5124, 9124, 2048),
#    (35, 8457, 2048),
#    (5124, 9124, 2560),
#    (35, 8457, 2560),
#    (5124, 9124, 4096),
#    (35, 8457, 4096),
#    (5124, 9124, 1760),
#    (35, 8457, 1760),
#    (5124, 9124, 2048),
#    (35, 8457, 2048),
#    (5124, 9124, 2560),
#    (35, 8457, 2560),
#    (5124, 9124, 4096),
#    (35, 8457, 4096),
#    (7680, 16, 2560),
#    (7680, 32, 2560),
#    (7680, 64, 2560),
#    (7680, 128, 2560),
#    (7680, 16, 2560),
#    (7680, 32, 2560),
#    (7680, 64, 2560),
#    (7680, 128, 2560),
#    (3072, 16, 1024),
#    (3072, 32, 1024),
#    (3072, 64, 1024),
#    (3072, 128, 1024),
#    (3072, 16, 1024),
#    (3072, 32, 1024),
#    (3072, 64, 1024),
#    (3072, 128, 1024),
#    (3072, 7435, 1024),
#    (7680, 5481, 2560),
#    (512, 8, 500000),
#    (1024, 8, 500000),
#    (512, 16, 500000),
#    (1024, 16, 500000),
#    (512, 8, 500000),
#    (1024, 8, 500000),
#    (512, 16, 500000),
#    (1024, 16, 500000),
#    (1024, 700, 512),
#    (1024, 700, 512),
#    (7680, 24000, 2560),
#    (6144, 24000, 2048),
#    (4608, 24000, 1536),
#    (8448, 24000, 2816),
#    (3072, 24000, 1024),
#    (7680, 48000, 2560),
#    (6144, 48000, 2048),
#    (4608, 48000, 1536),
#    (8448, 48000, 2816),
#    (3072, 48000, 1024),
#    (7680, 24000, 2560),
#    (6144, 24000, 2048),
#    (4608, 24000, 1536),
#    (8448, 24000, 2816),
#    (3072, 24000, 1024),
#    (7680, 48000, 2560),
#    (6144, 48000, 2048),
#    (4608, 48000, 1536),
#    (8448, 48000, 2816),
#    (3072, 48000, 1024),
#    (6144, 16, 2048),
#    (4608, 16, 1536),
#    (8448, 16, 2816),
#    (6144, 32, 2048),
#    (4608, 32, 1536),
#    (8448, 32, 2816),
#    (6144, 16, 2048),
#    (4608, 16, 1536),
#    (8448, 16, 2816),
#    (6144, 32, 2048),
#    (4608, 32, 1536),
#    (8448, 32, 2816),
#    (512, 24000, 2816),
#    (512, 24000, 2048),
#    (512, 24000, 2560),
#    (512, 24000, 1530),
#    (1024, 24000, 2816),
#    (1024, 24000, 2048),
#    (1024, 24000, 2560),
#    (1024, 24000, 1530),
#    (512, 16, 512),
#    (1024, 16, 512),
#    (512, 24000, 2816),
#    (512, 24000, 2048),
#    (512, 24000, 2560),
#    (512, 24000, 1530),
#    (1024, 24000, 2816),
#    (1024, 24000, 2048),
#    (1024, 24000, 2560),
#    (1024, 24000, 1530),
#    (512, 16, 512),
#    (1024, 16, 512),
#    (512, 48000, 2816),
#    (512, 48000, 2048),
#    (512, 48000, 2560),
#    (512, 48000, 1530),
#    (1024, 48000, 2816),
#    (1024, 48000, 2048),
#    (1024, 48000, 2560),
#    (1024, 48000, 1530),
#    (512, 32, 512),
#    (1024, 32, 512),
#    (512, 48000, 2816),
#    (512, 48000, 2048),
#    (512, 48000, 2560),
#    (512, 48000, 1530),
#    (1024, 48000, 2816),
#    (1024, 48000, 2048),
#    (1024, 48000, 2560),
#    (1024, 48000, 1530),
#    (512, 32, 512),
#    (1024, 32, 512),
# ]


matmul_tests = []


@initialize()
def initialize_tests():
    for M in [32, 64, 128, 2048, 4096]:
        for [N, K] in [[6144, 4096], [4096, 14336], [14336, 4096], [4096, 4096], [10240, 8192], [8192, 28672], [8192, 8192], [28672, 8192]]:
            matmul_tests.append((M, N, K, 1))
    return

    if bench_suites == "functional":
        matmul_tests.append((256, 256, 256, 1))
        matmul_tests.append((256, 256 + 2, 256, 1))
        matmul_tests.append((256, 256 - 4, 256, 1))
    elif bench_suites == "performance":
        matmul_tests.append((16384, 8, 8192, 1))
        matmul_tests.append((120000, 768, 320, 1))
        matmul_tests.append((120000, 320, 768, 1))
        matmul_tests.append((120000, 256, 320, 1))
        matmul_tests.append((120000, 320, 1024, 1))
        matmul_tests.append((120000, 1024, 320, 1))
        for m, n, k in dense_matmul:
            matmul_tests.append((m, n, k, 1))
        # LLMs
        for batchsize in [1, 16]:
            # Llama7B
            matmul_tests.append((batchsize, 4096 * 3, 4096, 1))
            matmul_tests.append((batchsize, 4096, 4096, 1))
            matmul_tests.append((batchsize, 2 * 10752, 4096, 1))
            matmul_tests.append((batchsize, 10752, 4096, 1))

            # Llama13B
            matmul_tests.append((batchsize, 5120, 3 * 5120, 1))
            matmul_tests.append((batchsize, 5120, 5120, 1))
            matmul_tests.append((batchsize, 5120, 2 * 13568, 1))
            matmul_tests.append((batchsize, 13568, 5120, 1))

            # Llama33B
            matmul_tests.append((batchsize, 6656, 3 * 6656, 1))
            matmul_tests.append((batchsize, 6656, 6656, 1))
            matmul_tests.append((batchsize, 6656, 2 * 17664, 1))
            matmul_tests.append((batchsize, 17664, 6656, 1))

            # Llama65B
            matmul_tests.append((batchsize, 8192, 3 * 8192, 1))
            matmul_tests.append((batchsize, 8192, 8192, 1))
            # skip because exceed 32-bit integer range, may cause illegal
            # memory access
            # matmul_tests.append((batchsize, 8192, 2 * 217600, 1))
            matmul_tests.append((batchsize, 217600, 8192, 1))
    else:
        raise NotImplementedError()


def data(M, N, K, L, dtype="bfloat16", device="cuda"):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    a = torch.randint(low=lo, high=hi, size=(L, M, K), dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=(L, K, N), dtype=dtype, device=device)

    return a, b


def test_problem(M, N, K, L, dtype):
    def graph(a, b):
        c = a @ b
        return c

    graph_args = data(M, N, K, L, dtype=dtype)

    import torch._dynamo as dynamo

    dynamo.reset()
    torch_mean, torch_min, torch_max = bench(graph, graph_args)
    print(f"baseline(torch.compile mode=max-autotune): {torch_mean} ms")

    with hidet.option.context():
        hidet.option.cache_dir(f"./matmul_standalone")
        hidet.option.debug_cache_tuning()
        hidet.option.save_lower_ir(True)
        hidet.option.parallel_k(strategy='disabled')
        hidet.option.search_space(2)
        hidet.option.hexcute_matmul(strategy='enable')

        def matmul_graph():
            a = hidet.symbol([L, M ,K], dtype=dtype, device='cuda')
            b = hidet.symbol([L, K, N], dtype=dtype, device='cuda')
            c = ops.matmul(a, b)
            return hidet.trace_from(c, [a, b])

        import time
 
        graph_hidet = matmul_graph()
        graph_hidet = hidet.graph.optimize(graph_hidet)
        D = graph(*graph_args)
        hidet_args = [hidet.from_torch(t) for t in graph_args]
        D_hidet = graph_hidet(*hidet_args)
        np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
        
        matmul_task = graph_hidet.get_compiled_task(0)
        D_hidet_ = matmul_task.create_outputs(hidet_args)[0]
        kernel = matmul_task.candidates[matmul_task.pick_best_candidate(hidet_args, [D_hidet_])]
        
        def hidet_matmul(a, b):
            kernel(a, b, D_hidet_)
            return D_hidet_

        D_hidet = D_hidet.torch()
        # change the tolarence because we use fp16 as accumulators
        np.testing.assert_allclose(
            actual=D_hidet.to(torch.float32).cpu().numpy(), desired=D.to(torch.float32).cpu().numpy(), rtol=5e-2
        )
        hidet_mean, hidet_min, hidet_max = bench(hidet_matmul, hidet_args)
        print(f"hidet(torch.compile): {hidet_mean} ms")

    from matmul import matmul as triton_matmul
    def triton_mm(a, b):
        a = a.view(a.shape[1], a.shape[2])
        b = b.view(b.shape[1], b.shape[2])
        return triton_matmul(a, b, torch.float32)
    triton_mean, triton_min, triton_max = bench(triton_mm, graph_args)
    print(f"triton: {triton_mean} ms")
    return torch_mean, triton_mean, hidet_mean


def main():
    from tabulate import tabulate

    records = []
    headers = ["problem(m,n,k,l)", "cublas", "triton", "hexcute", "flops_cublas", "flops_triton", "flops_hexcute"]
    cublas = []
    triton = []
    hexcute = []

    for problem in matmul_tests:
        cublas_time, triton_time, hexcute_time = test_problem(*problem, dtype='float16')
        M, N, K, L = problem
        flops = 2 * M * N * K
        flops_cublas = flops / cublas_time / 1e9
        flops_triton = flops / triton_time / 1e9
        flops_hexcute = flops / hexcute_time / 1e9
        records.append([problem, cublas_time, triton_time, hexcute_time, flops_cublas, flops_triton, flops_hexcute])
        cublas.append(flops_cublas)
        triton.append(flops_triton)
        hexcute.append(flops_hexcute)

    with open(f"matmul_a100.txt", "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )

    methods = ['Hexcute', 'Marlin-old', 'Marlin-new', 'Triton', 'Ladder', 'cuBLAS']
    methods = ['Hexcute', 'FlashAttention', 'FlashInfer', 'Triton', 'CUTLASS', 'cuBLAS']
    
    clist = ['#b5739d', '#7ea6e0', '#67ab9f', '#ea6b66', '#ffb570', '#97d077']
    #clist = ['#38761D', '#4285F4', '#EA4335', '#ea6b66', '#ffb570', '#97d077']
    my_colors = {}
    for i, method in enumerate(methods):
        my_colors[method] = clist[i]

    import matplotlib.pyplot as plt
    from matplotlib import rc     
    rc('font', **{'family': 'sans-serif', 'size': 25})
    import numpy as np

    # Data for each method
    methods = ['Triton', 'cuBLAS', 'Hexcute']

    fig, ax = plt.subplots(1, 1, figsize=(30, 4))

    print(len(triton))
    categories = [f"M{m}" for m in range(len(cublas))]
#    categories = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, '2K', '4K', '8K', '16K']
    # categories = [1, 8, 16, 32, 64, 128, 256]
    N = len(categories)
    ind = np.arange(N)  # X locations for the groups
    width = 0.2         # Width of the bars

    import numpy as np
    cmap = plt.get_cmap('gnuplot')
    ll = cmap.N*8//9
    len_methods = len(methods)
    indices = np.linspace(ll//5, ll, len_methods)
    #my_colors = [cmap(int(i)) for i in indices]
 
    # Plotting the bars for each method across matrix types (speedup)
#    print(len(speedup_marlin), len(speedup_tri), len(speedup_hi))
    gap = 0.012
    i = 0
    ax.bar(ind + (i + 0.5) * (width + gap), triton, width, label=methods[i], color=my_colors[methods[i]])
    i = 1
    ax.bar(ind + (i + 0.5) * (width + gap), cublas, width, label=methods[i], color=my_colors[methods[i]])
    i = 2
    ax.bar(ind + (i + 0.5) * (width + gap), hexcute, width, label=methods[i], color=my_colors[methods[i]])
 
    #v = ind[-1] + 2.5 * (width + 0.2) * 0.6
    #ax.text(v, speedup_hi[-1], f'{1 / speedup_tri[-1]:.2f}x', ha='center', va='bottom', fontsize=16)
    #ax.axhline(y=1, color='b', linestyle='--', linewidth=2)

 #   x = marlin_old
 #   for i in range(len(marlin_old)):
 #       if x[i] >= 14:
 #           ax.text(ind[i] + 0.5 * (width + gap), 14, f'{marlin_old[i]:.0f}', ha='center', va='bottom', fontsize=10, color='black')

 #   x = triton
 #   for i in range(len(marlin_old)):
 #       if x[i] >= 14:
 #           ax.text(ind[i] + (1 + 0.5) * (width + gap), 13, f'{x[i]:.0f}', ha='center', va='bottom', fontsize=10, color='black')

    max_y = max(max(triton), max(cublas), max(hexcute)) + 25
    ax.set_ylabel('Throughput (TFLOPS)', fontsize=18)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('F16 GEMM Layers', fontsize=18)
    ax.set_xticks(ind + (len(methods) * width) / 2)
    ax.set_yticks(np.arange(0, max_y, 25))
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    ax.set_xticklabels(categories, fontsize=18)
    # title_loc = -0.2
    #ax.set_title('FP16xINT4 MoE Layer', fontsize=18)
    ax.yaxis.grid(True, linestyle='dotted')

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    x = set()
    lins = []
    labs = []
    for li, la in zip(lines, labels):
        if la in x:
            continue
        x.add(la)
        lins.append(li)
        labs.append(la)
    fig.legend(lins, labs, loc='upper left', bbox_to_anchor=(0.048, 0.95), fontsize=14, ncols=3)

    fig.subplots_adjust(
            top=0.94,
            bottom=0.173,
            left=0.048,
            right=0.99,
            hspace=0.2,
            wspace=0.2
        )
    # Adjust layout to prevent clipping of tick-labels
    plt.savefig("matmul_a100.pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
