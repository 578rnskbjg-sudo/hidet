# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas.utils.target_detector import auto_detect_nvidia_target
from bitblas import Matmul, MatmulConfig
import argparse
import torch


# Initialize the parser
parser = argparse.ArgumentParser(
    description="Benchmark BitBLAS int4 on a specific target."
)

# Add arguments to the parser
parser.add_argument(
    "--target",
    type=str,
    default=auto_detect_nvidia_target(),
    help="Specify the target device for benchmarking."
)
parser.add_argument(
    "--group_size",
    type=int,
    default=128,
    help="Group size for grouped quantization."
)
parser.add_argument(
    "--A_dtype",
    type=str,
    default="float16",
    choices=["float16", "float32", "float64", "int32", "int8", "int4"],  # Assuming these are the valid choices
    help="Data type of activation A."
)
parser.add_argument(
    "--W_dtype",
    type=str,
    default="int4",
    choices=["float16", "float32", "float64", "int32", "int8", "int4", "int2", "int1", "nf4", "fp4_e2m1", "uint1"],  # Assuming these are the valid choices
    help="Data type of weight W."
)
parser.add_argument(
    "--accum_dtype",
    type=str,
    default="float16",
    choices=["float16", "int32"],  # Assuming these are the valid choices
    help="Data type for accumulation."
)
parser.add_argument(
    "--out_dtype",
    type=str,
    default="float16",
    choices=["float16", "float32", "int32", "int8"],  # Assuming these are the valid choices
    help="Data type for output."
)
parser.add_argument(
    "--layout",
    type=str,
    default="nt",
    choices=["nt", "nn"],  # Assuming these are the valid choices
    help="Matrix layout, 'nt' for non-transpose A and transpose W."
)
parser.add_argument(
    "--with_bias",
    action="store_true",
    help="Include bias in the benchmark."
)
parser.add_argument(
    "--with_scaling",
    action="store_true",
    help="Include scaling factor in the quantization."
)
parser.add_argument(
    "--with_zeros",
    action="store_true",
    help="Include zeros in the quantization."
)
parser.add_argument(
    "--zeros_mode",
    type=str,
    default=None,
    choices=["original", "rescale", "quantized"],  # Replace with actual modes if applicable
    help="Specify the mode for calculating zeros."
)

# Parse the arguments
args = parser.parse_args()

# Assign arguments to variables
target = args.target
group_size = args.group_size
A_dtype = args.A_dtype
W_dtype = args.W_dtype
accum_dtype = args.accum_dtype
out_dtype = args.out_dtype
layout = args.layout
with_bias = args.with_bias
group_size = args.group_size
with_scaling = args.with_scaling
with_zeros = args.with_zeros
zeros_mode = args.zeros_mode

test_shapes = [
    #(MatmulConfig, Matmul, (16, 4096*3, 4096, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    #(MatmulConfig, Matmul, (8, 6656, 6656, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    #(MatmulConfig, Matmul, (8, 6656, 2 * 17664, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    #(MatmulConfig, Matmul, (8, 17664, 6656, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),

    ## # LLAMA-70B/65B
    (MatmulConfig, Matmul, (1, 8192, 8192, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    (MatmulConfig, Matmul, (1, 28672, 8192, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    (MatmulConfig, Matmul, (1, 8192, 28672, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    (MatmulConfig, Matmul, (8, 8192, 8192, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    (MatmulConfig, Matmul, (8, 28672, 8192, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    (MatmulConfig, Matmul, (8, 8192, 28672, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    #(MatmulConfig, Matmul, (16, 8192, 8192, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    #(MatmulConfig, Matmul, (16, 28672, 8192, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    #(MatmulConfig, Matmul, (16, 8192, 28672, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    #(MatmulConfig, Matmul, (32, 8192, 8192, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    #(MatmulConfig, Matmul, (32, 28672, 8192, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),
    #(MatmulConfig, Matmul, (32, 8192, 28672, A_dtype, W_dtype, out_dtype, accum_dtype, layout, with_bias, group_size, with_scaling, with_zeros, zeros_mode)),

]



def data(M, N, K, weight_bits=4, group_size=64, dtype="float16", device="cuda"):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    a = torch.randint(low=lo, high=hi, size=(M, K), dtype=dtype, device=device)
    b = torch.randint(low=0, high=hi, size=(N, K), dtype=torch.int8, device=device)
    scale = torch.randint(low=-1, high=2, size=(K // group_size, N), dtype=dtype, device=device)
    zeros = torch.randint(low=0, high=hi, size=(K // group_size, N), dtype=dtype, device=device)

    ret = [a, b, scale, zeros]
    return tuple(ret)


def bench(f, warmup=5, number=1, repeat=31):
    import numpy as np

    cache = torch.empty((int(256e6),), dtype=torch.int8, device='cuda')
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(repeat)]

    for _ in range(warmup):
        cache.zero_()
        f()

    for i in range(repeat):
        cache.zero_()
        start_event[i].record()
        f()
        end_event[i].record()
    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    return np.mean(times).item()


benchmark_sets = []
benchmark_sets.extend(test_shapes)

# fmt:on

benchmark_results = {}
for config, operator, input_args in benchmark_sets:
    config = config(*input_args)
    matmul = operator(config, target=target, enable_tuning=True)
    print(matmul.get_source())
    #bq4 = matmul.transform_weight(b)
    #bias_q4 = matmul.transform_weight(bias)
    #if config.A_dtype == "int8":
    #    a = a.to(torch.int8)
    #if config.B_dtype == "int8":
    #    b = b.to(torch.int8)
    weigth_bits = 4
    if config.W_dtype == "int1":
        weight_bits = 1
    elif config.W_dtype == "int2":
        weight_bits = 2
    elif config.W_dtype == "int4":
        weight_bits = 4
    elif config.W_dtype == "int8":
        weight_bits = 8
    a, b, scale, bias = data(config.M, config.N, config.K, dtype="int8", group_size=config.group_size)
    b = matmul.transform_weight(b)

    def f():
        if config.with_scaling and config.with_zeros:
            c = matmul(a, b, scale, bias)
        elif config.with_scaling:
            c = matmul(a, b, scale)
        else:
            c = matmul(a, b)
        return c
    kernel_latency = bench(f, warmup=10, repeat=100)
    print("==================================")
    m = config.M
    n = config.N
    k = config.K
    print("{}x{}x{}".format(config.M, config.N, config.K))
    print("Time cost is: {:.3f} ms".format(kernel_latency))
    mean = kernel_latency
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
    print(
        "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
            mean, (m * n * 2 + m * k * 2 + k * n // 2) / (1e6 * mean)
        )
    )
    #print(matmul.get_source())

    profile_config = {
        f"{operator.__name__}-{'-'.join([str(i) for i in input_args])}": {
            "BitBLAS_top20_latency": kernel_latency,
        }
    }

    benchmark_results.update(profile_config)

# Define headers for the table
headers = [
    "PrimFunc",
    "Input Arguments",
    "BitBLAS Top20 Latency",
]

col_widths = [0, 0, 0]
for config, values in benchmark_results.items():
    args = config.split("-")
    func_name = args[0]
    input_args = "-".join(args[1:])
    col_widths[0] = max((max(len(str(headers[0])), len(func_name)) + 2), col_widths[0])
    col_widths[1] = max((max(len(str(headers[1])), len(input_args)) + 2, col_widths[1]))
    col_widths[2] = max(max(len(str(headers[2])), len(f"{values['BitBLAS_top20_latency']:.3f} ms")) + 2, col_widths[2])
    break

for i, header in enumerate(headers):
    headers[i] = header.ljust(col_widths[i])

print("".join(headers))

print("-" * sum(col_widths))

for config, values in benchmark_results.items():
    args = config.split("-")
    func_name = args[0]
    input_args = "-".join(args[1:])
    row = [
        func_name,
        input_args,
        f"{values['BitBLAS_top20_latency']:.3f} ms",
    ]
    print("".join([str(i).ljust(col_widths[j]) for j, i in enumerate(row)]) + "\n")
