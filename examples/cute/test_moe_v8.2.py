from moe_wna16 import fused_moe_wna16, preprocess_weight
from moe_utils import cast_f16_to_u4, cast_u4_to_f16
from typing import Union, List, Optional, Callable, Any
import functools
import itertools
import pytest

import torch
import hidet

from hidet.ir.dtypes import bf16, f16, u4, f32, i32, i64
from hidet.ir.type import DataType

from hidet.graph.frontend.torch.utils import dtype_to_torch
from hidet.logging import logger, setConsoleLevel, INFO
from hidet.utils.benchmark import do_bench


def create_parameters(
    e: int, k: int, n: int, group_size: int, param_dtype: Union[str, DataType], act_dtype: Union[str, DataType] = "float16"
):
    bdtype = dtype_to_torch(act_dtype)
    scale_dtype = dtype_to_torch(act_dtype)
    bias_dtype = dtype_to_torch(act_dtype)
    device = "cuda"

    lo = 0
    hi = 3
    weight = torch.randint(low=0, high=hi, size=(e, k, n), dtype=bdtype, device=device)
    scale = torch.randint(low=-2, high=2, size=(e, k // group_size, n), dtype=scale_dtype, device=device)
    zeros = torch.randint(low=0, high=hi, size=(e, k // group_size, n), dtype=bias_dtype, device=device)
    qweight = preprocess_weight(cast_f16_to_u4(weight.reshape(-1, n)).reshape(e, k, n // 2))
    qweight = qweight
    return weight, qweight, scale, zeros


def weight_to_triton_weight(weight, scale, zeros, e, k, n, group_size, act_dtype: Union[str, DataType] = "float16"):
    triton_weight = weight.permute(0, 2, 1).reshape(-1, k)
    triton_qweight = cast_f16_to_u4(triton_weight, act_dtype).reshape(e, n, k // 2)
    triton_scale = scale.permute(0, 2, 1)
    triton_zeros = zeros.reshape(e, k // group_size, n // 2, 2).permute(0, 2, 1, 3).reshape(-1, k // group_size * 2)
    triton_qzeros = cast_f16_to_u4(triton_zeros, act_dtype).reshape(e, n // 2, k // group_size)

    return triton_qweight, triton_scale, triton_qzeros


def triton_weight_to_weight(qweight, scale, qzeros, e, k, n, group_size, act_dtype: Union[str, DataType] = "float16"):
    weight = cast_u4_to_f16(qweight.reshape(-1, k // 2), act_dtype)
    weight = weight.reshape(e, n, k).permute(0, 2, 1)
    scale = scale.permute(0, 2, 1)
    zeros = cast_u4_to_f16(qzeros.reshape(-1, k // group_size), act_dtype)
    zeros = zeros.reshape(e, n // 2, k // group_size, 2).permute(0, 2, 1, 3).reshape(e, k // group_size, n)
    return weight, scale, zeros


verbose = False


@pytest.mark.parametrize("tokens", [[1, 32, 222]])
@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("experts_per_token", [2, 6])
@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("param_dtype", [u4])
@pytest.mark.parametrize("act_dtype", [f16])
def test_fused_moe_wna16(
    tokens: List[int],
    k: int,
    n: int,
    experts_per_token: int,
    num_experts: int,
    group_size: int,
    param_dtype: DataType,
    act_dtype: DataType,
    cache_dir: str = "fused_moe",
    triton_dataflow: bool = False,
    triton_shared_memory: bool = False,
    output: str = "fused_moe.txt",
    verify_result: bool = True,
):
    with hidet.option.context():
        hidet.option.cache_dir(cache_dir)
        hidet.option.debug_cache_tuning(True)
        hidet.option.save_lower_ir(True)
        hidet.option.search_space(2)
#        hidet.option.num_local_workers(1)
        hidet.option.use_torch_stream(True)
        setConsoleLevel(INFO)

        moe = fused_moe_wna16(k, n, experts_per_token, num_experts, group_size, param_dtype, act_dtype, triton_dataflow, triton_shared_memory)

    hidet.option.use_torch_stream(True)
    import numpy as np
    from tabulate import tabulate
    records = []
    headers = ["num_tokens", "triton", "marlin", "hexcute"]
    
    for num_tokens in tokens:
        renormalize = False
        adtype = dtype_to_torch(act_dtype)

        n2 = n * 2
        lo = -3
        hi = 3
        hidden_state = torch.randint(low=lo, high=hi, size=(num_tokens, k), dtype=adtype, device="cuda") / np.sqrt(k)
        weight1, qweight1, scales1, zeros1 = create_parameters(num_experts, k, n2, group_size, param_dtype, act_dtype)
        weight2, qweight2, scales2, zeros2 = create_parameters(num_experts, n, k, group_size, param_dtype, act_dtype)

        score = torch.randn((num_tokens, num_experts), device="cuda", dtype=adtype)

        from vllm.model_executor.layers.fused_moe import fused_topk

        topk_weights, topk_ids = fused_topk(hidden_state, score, experts_per_token, renormalize=renormalize)

        out_hidden_state, intermediate_cache1, intermediate_cache2, intermediate_cache3 = moe(
            hidden_state,
            qweight1,
            scales1,
            zeros1,
            qweight2,
            scales2,
            zeros2,
            topk_ids,
            topk_weights,
            return_intermediate_caches=True,
        )
        print(intermediate_cache1)
        print(intermediate_cache2)
        print(intermediate_cache3)
        print(out_hidden_state)

        def fn():
            topk_weights, topk_ids = fused_topk(hidden_state, score, experts_per_token, renormalize=renormalize)
            return moe(
                hidden_state,
                qweight1,
                scales1,
                zeros1,
                qweight2,
                scales2,
                zeros2,
                topk_ids,
                topk_weights,
                return_intermediate_caches=False,
            )

        torch.cuda.profiler.cudart().cudaProfilerStart()
        time_hexcute = do_bench(fn, percentiles=None)
        provider = "hexcute"
        print(f"k: {k}, n: {n}")
        print(f"experts_per_token: {experts_per_token}, num_experts: {num_experts}, group_size: {group_size}")
        print(f"provider: {provider}, num_tokens: {num_tokens}, time: {time_hexcute}")

        del qweight1
        del qweight2
        try:
            from vllm.model_executor.layers.fused_moe import fused_moe

            e = num_experts
            topk = experts_per_token
            m = num_tokens
            weight_bits = 4
            has_zp = True
            w1_qweight, w1_scales, w1_qzeros = weight_to_triton_weight(weight1, scales1, zeros1, e, k, n2, group_size, act_dtype)
            w2_qweight, w2_scales, w2_qzeros = weight_to_triton_weight(weight2, scales2, zeros2, e, n, k, group_size, act_dtype)
            if verbose:
                w1, s1, z1 = triton_weight_to_weight(w1_qweight, w1_scales, w1_qzeros, e, k, n2, group_size, act_dtype)
                w2, s2, z2 = triton_weight_to_weight(w2_qweight, w2_scales, w2_qzeros, e, n, k, group_size, act_dtype)
                np.testing.assert_allclose(w1.cpu(), weight1.cpu(), rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(w2.cpu(), weight2.cpu(), rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(s1.cpu(), scales1.cpu(), rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(s2.cpu(), scales2.cpu(), rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(z1.cpu(), zeros1.cpu(), rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(z2.cpu(), zeros2.cpu(), rtol=1e-3, atol=1e-3)

            triton_output = fused_moe(
                hidden_state,   
                w1_qweight,
                w2_qweight,
                score,
                topk,
                renormalize=False,
                use_int4_w4a16=weight_bits == 4,
                use_int8_w8a16=weight_bits == 8,
                global_num_experts=num_experts,
                expert_map=None,
                w1_scale=w1_scales,
                w2_scale=w2_scales,
                w1_zp=w1_qzeros if has_zp else None,
                w2_zp=w2_qzeros if has_zp else None,
                block_shape=[0, group_size],
            )
            print(triton_output)

            def fn():
                return fused_moe(
                    hidden_state,
                    w1_qweight,
                    w2_qweight,
                    score,
                    topk,
                    renormalize=False,
                    use_int4_w4a16=weight_bits == 4,
                    use_int8_w8a16=weight_bits == 8,
                    global_num_experts=e,
                    expert_map=None,
                    w1_scale=w1_scales,
                    w2_scale=w2_scales,
                    w1_zp=w1_qzeros if has_zp else None,
                    w2_zp=w2_qzeros if has_zp else None,
                    block_shape=[0, group_size],
                )

            time_triton = do_bench(fn, percentiles=None)
            provider = "triton"
            print(f"k: {k}, n: {n}")
            print(f"experts_per_token: {experts_per_token}, num_experts: {num_experts}, group_size: {group_size}")
            print(f"provider: {provider}, num_tokens: {num_tokens}, time: {time_triton}")
            rtol = 1e-2 if act_dtype == f16 else 5e-2
            if verify_result:
                np.testing.assert_allclose(triton_output.to(torch.float32).cpu(), out_hidden_state.to(torch.float32).cpu(), rtol=rtol, atol=1)

            del w1_qweight
            del w2_qweight
            del w1_scales
            del w2_scales
            del w1_qzeros
            del w2_qzeros

            from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
                awq_marlin_quantize)
            from vllm.scalar_type import scalar_types

            def stack_and_dev(tensors: List[torch.Tensor]):
                dev = tensors[0].device
                return torch.stack(tensors, dim=0).to(dev)

            quant_type = scalar_types.uint4b8
            #quant_type = scalar_types.uint4

            def create_weights_for_marlin():
                qweight1_l = []
                scales1_l = []
                zeros1_l = []

                for i in range(weight1.shape[0]):
                    w_ref1, qweight1, scales1, zeros1 = awq_marlin_quantize(
                        weight1[i], quant_type, group_size)
                    qweight1_l.append(qweight1)
                    scales1_l.append(scales1)
                    zeros1_l.append(zeros1)

                qweight1 = stack_and_dev(qweight1_l).contiguous()
                scales1 = stack_and_dev(scales1_l)
                zeros1 = stack_and_dev(zeros1_l)

                qweight2_l = []
                scales2_l = []
                zeros2_l = []

                for i in range(weight2.shape[0]):
                    w_ref2, qweight2, scales2, zeros2 = awq_marlin_quantize(
                        weight2[i], quant_type, group_size)
                    qweight2_l.append(qweight2)
                    scales2_l.append(scales2)
                    zeros2_l.append(zeros2)

                qweight2 = stack_and_dev(qweight2_l).contiguous()
                scales2 = stack_and_dev(scales2_l)
                zeros2 = stack_and_dev(zeros2_l)
                return qweight1, scales1, zeros1, qweight2, scales2, zeros2

            w1_qweight, w1_scales, w1_zeros, w2_qweight, w2_scales, w2_zeros = create_weights_for_marlin()
            w1_qweight.random_(0, 3)
            w2_qweight.random_(0, 3)
            w1_scales.random_(-2, 2)
            w2_scales.random_(-2, 2)
            w1_zeros.random_(0, 3)
            w2_zeros.random_(0, 3)
           
            def fn():
                topk_weights, topk_ids = fused_topk(hidden_state, score, topk, renormalize=renormalize)
                marlin_output = torch.ops.vllm.fused_marlin_moe(
                    hidden_state,
                    w1_qweight,
                    w2_qweight,
                    w1_scales,
                    w2_scales,
                    score,
                    topk_weights,
                    topk_ids,
                    #quant_type_id=quant_type.id,
                    w1_zeros=w1_zeros,
                    w2_zeros=w2_zeros,
                    num_bits=4,
                )
                return marlin_output

            time_marlin = do_bench(fn, percentiles=None)
            provider = "marlin"
            print(f"k: {k}, n: {n}")
            print(f"experts_per_token: {experts_per_token}, num_experts: {num_experts}, group_size: {group_size}")
            print(f"provider: {provider}, num_tokens: {num_tokens}, time: {time_marlin}")
            records.append([num_tokens, time_triton, time_marlin, time_hexcute])
        except ImportError:
            logger.info("vllm not installed")
        torch.cuda.profiler.cudart().cudaProfilerStop()
    with open(output, "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )


def markdown_table_to_dicts(markdown_file):
    """
    Converts a Markdown table string into a list of dictionaries.
    Each dictionary represents a row, with keys being the column headers.
    """
    with open(markdown_file, "r") as f:
        markdown_table_string = f.read()
    lines = markdown_table_string.strip().split('\n')
    if len(lines) < 2:
        return []  # Not a valid table (needs at least header and separator)

    # Extract headers
    headers = [h.strip() for h in lines[0].split('|') if h.strip()]

    # Skip the separator line (lines[1])
    data_rows = lines[2:]
    header_values = [[] for _ in headers]

    for row_str in data_rows:
        values = [v.strip() for v in row_str.split('|') if v.strip()]
        if len(values) == len(headers):
            for value, header_value in zip(values, header_values):
                header_value.append(value)
    return header_values


def generate_performance_plot(triton, hexcute, marlin_old, marlin_new):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    triton = [float(x) for x in triton]
    hexcute = [float(x) for x in hexcute]
    marlin_old = [float(x) for x in marlin_old]
    marlin_new = [float(x) for x in marlin_new]
    methods = ['Hexcute', 'Marlin-old', 'Marlin-new', 'Triton', 'Ladder', 'cuBLAS']
    
    clist = ['#b5739d', '#7ea6e0', '#67ab9f', '#ea6b66', '#ffb570', '#97d077']
    #clist = ['#38761D', '#4285F4', '#EA4335', '#ea6b66', '#ffb570', '#97d077']
    my_colors = {}
    for i, method in enumerate(methods):
        my_colors[method] = clist[i]
    rc('font', **{'family': 'sans-serif', 'size': 25})
    import numpy as np

    # Data for each method
    methods = ['Marlin-old', 'Triton', 'Marlin-new', 'Hexcute']

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    categories = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, '2K', '4K', '8K', '16K']
    # categories = [1, 8, 16, 32, 64, 128, 256]
    N = len(categories)
    ind = np.arange(N)  # X locations for the groups
    width = 0.22         # Width of the bars

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
    ax.bar(ind + (i + 0.5) * (width + gap), marlin_old, width, label=methods[i], color=my_colors[methods[i]])
    i = 1
    ax.bar(ind + (i + 0.5) * (width + gap), triton, width, label=methods[i], color=my_colors[methods[i]])
    i = 2
    ax.bar(ind + (i + 0.5) * (width + gap), marlin_new, width, label=methods[i], color=my_colors[methods[i]])
    i = 3
    ax.bar(ind + (i + 0.5) * (width + gap), hexcute, width, label=methods[i], color=my_colors[methods[i]])
 
    #v = ind[-1] + 2.5 * (width + 0.2) * 0.6
    #ax.text(v, speedup_hi[-1], f'{1 / speedup_tri[-1]:.2f}x', ha='center', va='bottom', fontsize=16)
    #ax.axhline(y=1, color='b', linestyle='--', linewidth=2)

    x = marlin_old
    for i in range(len(marlin_old)):
        if x[i] >= 14:
            ax.text(ind[i] + 0.5 * (width + gap), 14, f'{marlin_old[i]:.0f}', ha='center', va='bottom', fontsize=10, color='black')

    x = triton
    for i in range(len(marlin_old)):
        if x[i] >= 14:
            ax.text(ind[i] + (1 + 0.5) * (width + gap), 13, f'{x[i]:.0f}', ha='center', va='bottom', fontsize=10, color='black')

    ax.set_ylabel('Latency (ms)', fontsize=18)
    ax.set_ylim(0, 14)
    ax.set_xlabel('Number of Tokens', fontsize=18)
    ax.set_xticks(ind + (len(methods) * width) / 2)
    ax.set_yticks(np.arange(0, 14, 2))
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
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
    fig.legend(lins, labs, loc='upper left', bbox_to_anchor=(0.09, 0.945), fontsize=14, ncols=1)

    fig.subplots_adjust(
            top=0.94,
            bottom=0.205,
            left=0.096,
            right=0.99,
            hspace=0.2,
            wspace=0.2
        )
    # Adjust layout to prevent clipping of tick-labels
    plt.savefig("moewna16_performance_plot.pdf", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    test_fused_moe_wna16([1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, 2048, 4096, 8192, 16384], 7168, 256, 8, 256, 64, u4, f16, output="fused_moe_v8.2.txt")
   
    _, _, marlin_old, hexcute = markdown_table_to_dicts("fused_moe_v8.2.txt")
    _, triton, marlin_new, hexcute = markdown_table_to_dicts("fused_moe.txt")
    generate_performance_plot(triton, hexcute, marlin_old, marlin_new)