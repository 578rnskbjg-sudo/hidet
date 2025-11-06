import pytest
import argparse

import hidet
from gemm_quant_intxb import IntxbAIntxbBGemm
from hidet.utils.py import cdiv

import torch

from quant_utils import bench, data_intxb


def bench_int8bint8b():
    hidet.option.cache_dir("./demo_gemm_quant_int8bint8b")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    problem_sizes = [
        (1, 8192, 8192),
        (1, 28672, 8192),
        (1, 8192, 28672),
        (8, 8192, 8192),
        (8, 28672, 8192),
        (8, 8192, 28672),
        (16, 8192, 8192),
        (16, 28672, 8192),
        (16, 8192, 28672),
        (32, 8192, 8192),
        (32, 28672, 8192),
        (32, 8192, 28672),
    ]

    from tabulate import tabulate

    records = []
    headers = ["problem(m,n,k)", "latency", "idx"]

    scale = 0.01
    for m, n, k in problem_sizes:
        print(f"m, n, k: {m}, {n}, {k}")
        a, b, c = data_intxb(m, n, k, return_hidet=True)

        linear = IntxbAIntxbBGemm(m, k, n, 8, 8, scale=scale, out_dtype=hidet.dtypes.i8)

        best_time = None
        best_module = None
        best_idx = None
        modules = linear.modules()
        for i, quant_module in enumerate(modules):
            module = quant_module.gemm_quant

            func = module.build()

            tiled_mma = quant_module._tuning_kwargs["tiled_mma"]
            parallel_k_parts = quant_module._tuning_kwargs["parallel_k_parts"]
            print(quant_module._tuning_kwargs)
            c_shape, _ = tiled_mma.c_tv_layout()
            block_m, block_n = c_shape
            gridm = cdiv(m, block_m)
            gridn = cdiv(n, block_n)

            def fn():
                c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float32, device="cuda")
                c = torch.empty((m, n), dtype=torch.int8, device="cuda")
                counters = torch.empty((gridm, gridn), dtype=torch.int32, device="cuda")
                func(a, b, c, c_parallel_k_parts, counters)
                return c

            mean, min_lat, max_lat = bench(fn, ())
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            from hidet.utils.benchmark import do_bench

            mean = do_bench(fn, percentiles=None)
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            if best_time is None:
                best_time = mean
                best_module = module
                best_idx = i
            elif mean < best_time:
                best_time = mean
                best_module = module
                best_idx = i

            c = fn()
            c2 = a.torch().to(torch.float16) @ b.torch().to(torch.float16).T
            c2 = (c2 * scale).to(torch.int8)
            import numpy as np

            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), atol=1, rtol=1e-2)

        #a = a.torch()
        #b = b.torch()

        #def fn():
        #    return a @ b.T
        #mean = do_bench(fn, percentiles=None)
        #print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
        #print(
        #    "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
        #        mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
        #    )
        #)
        print(f"m, n, k: {m}, {n}, {k}")
        print(best_time)
        print(best_module)
        print(best_idx)
        records.append([(m, n, k), best_time, best_idx])

    with open("results_quant_gemm_int8bint8b.txt", "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )


def bench_int8bint4b():
    hidet.option.cache_dir("./demo_gemm_quant_int4bxint8b")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    problem_sizes = [
        (1, 8192, 8192),
        (1, 28672, 8192),
        (1, 8192, 28672),
        (8, 8192, 8192),
        (8, 28672, 8192),
        (8, 8192, 28672),
        (16, 8192, 8192),
        (16, 28672, 8192),
        (16, 8192, 28672),
        (32, 8192, 8192),
        (32, 28672, 8192),
        (32, 8192, 28672),
    ]

    from tabulate import tabulate

    records = []
    headers = ["problem(m,n,k)", "latency", "idx"]

    scale = 0.01
    for m, n, k in problem_sizes:
        print(f"m, n, k: {m}, {n}, {k}")
        a, b, c = data_intxb(m, n, k, return_hidet=True)
        b_f16 = b.torch().T.contiguous().to(torch.float16)
        bq = torch.empty((k, n // 2), dtype=torch.uint8, device="cuda")

        linear = IntxbAIntxbBGemm(m, k, n, 8, 4, scale=scale, out_dtype=hidet.dtypes.i8)

        best_time = None
        best_module = None
        best_idx = None
        modules = linear.modules()
        for i, quant_module in enumerate(modules):
            module = quant_module.gemm_quant
            qfunc = quant_module.quant.build()
            dqfunc = quant_module.dequant.build()

            bdq = torch.empty((k, n), dtype=torch.float16, device="cuda")
            qfunc(b_f16, bq)
            dqfunc(bq, bdq)

            func = module.build()

            tiled_mma = quant_module._tuning_kwargs["tiled_mma"]
            parallel_k_parts = quant_module._tuning_kwargs["parallel_k_parts"]
            print(quant_module._tuning_kwargs)
            c_shape, _ = tiled_mma.c_tv_layout()
            block_m, block_n = c_shape
            gridm = cdiv(m, block_m)
            gridn = cdiv(n, block_n)

            def fn():
                c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float32, device="cuda")
                c = torch.empty((m, n), dtype=torch.int8, device="cuda")
                counters = torch.empty((gridm, gridn), dtype=torch.int32, device="cuda")
                func(a, bq, c, c_parallel_k_parts, counters)
                return c

            mean, min_lat, max_lat = bench(fn, ())
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            from hidet.utils.benchmark import do_bench

            mean = do_bench(fn, percentiles=None)
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            if best_time is None:
                best_time = mean
                best_module = module
                best_idx = i
            elif mean < best_time:
                best_time = mean
                best_module = module
                best_idx = i

            c = fn()
            c2 = a.torch().to(torch.float16) @ b.torch().to(torch.float16).T
            c2 = (c2 * scale).to(torch.int8)

            import numpy as np
            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), atol=1, rtol=5e-3)

        print(f"m, n, k: {m}, {n}, {k}")
        print(best_time)
        print(best_module)
        print(best_idx)
        records.append([(m, n, k), best_time, best_idx])

    with open("results_quant_gemm_int8bint4b_1.txt", "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )


def bench_int8bint2b():
    hidet.option.cache_dir("./demo_gemm_quant_int2bxint8b")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    problem_sizes = [
        #(1, 8192, 8192),
        #(1, 28672, 8192),
        #(1, 8192, 28672),
        #(8, 8192, 8192),
        #(8, 28672, 8192),
        #(8, 8192, 28672),
        #(16, 8192, 8192),
        (16, 28672, 8192),
        #(16, 8192, 28672),
        #(32, 8192, 8192),
        #(32, 28672, 8192),
        #(32, 8192, 28672),
    ]

    from tabulate import tabulate

    records = []
    headers = ["problem(m,n,k)", "latency", "idx"]

    scale = 0.01
    for m, n, k in problem_sizes:
        print(f"m, n, k: {m}, {n}, {k}")
        a, b, c = data_intxb(m, n, k, return_hidet=True)
        b_f16 = b.torch().T.contiguous().to(torch.float16)
        bq = torch.empty((k, n // 4), dtype=torch.uint8, device="cuda")

        linear = IntxbAIntxbBGemm(m, k, n, 8, 2, scale=scale, out_dtype=hidet.dtypes.i8)

        best_time = None
        best_module = None
        best_idx = None
        modules = linear.modules()
        for i, quant_module in enumerate(modules):
            module = quant_module.gemm_quant
            qfunc = quant_module.quant.build()
            dqfunc = quant_module.dequant.build()

            bdq = torch.empty((k, n), dtype=torch.float16, device="cuda")
            qfunc(b_f16, bq)
            dqfunc(bq, bdq)

            func = module.build()

            tiled_mma = quant_module._tuning_kwargs["tiled_mma"]
            parallel_k_parts = quant_module._tuning_kwargs["parallel_k_parts"]
            print(quant_module._tuning_kwargs)
            c_shape, _ = tiled_mma.c_tv_layout()
            block_m, block_n = c_shape
            gridm = cdiv(m, block_m)
            gridn = cdiv(n, block_n)

            def fn():
                c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float32, device="cuda")
                c = torch.empty((m, n), dtype=torch.int8, device="cuda")
                counters = torch.empty((gridm, gridn), dtype=torch.int32, device="cuda")
                func(a, bq, c, c_parallel_k_parts, counters)
                return c

            mean, min_lat, max_lat = bench(fn, ())
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            from hidet.utils.benchmark import do_bench

            mean = do_bench(fn, percentiles=None)
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            if best_time is None:
                best_time = mean
                best_module = module
                best_idx = i
            elif mean < best_time:
                best_time = mean
                best_module = module
                best_idx = i

            c = fn()
            c2 = a.torch().to(torch.float16) @ b.torch().to(torch.float16).T
            c2 = (c2 * scale).to(torch.int8)

            import numpy as np
            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), atol=1, rtol=5e-3)

        #a = a.torch()
        #b = b.torch()

        #def fn():
        #    return a @ b.T
        #mean = do_bench(fn, percentiles=None)
        #print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
        #print(
        #    "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
        #        mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
        #    )
        #)
        print(f"m, n, k: {m}, {n}, {k}")
        print(best_time)
        print(best_module)
        print(best_idx)
        records.append([(m, n, k), best_time, best_idx])

    with open("results_quant_gemm_int8bint2b.txt", "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )


def bench_int8bint1b():
    hidet.option.cache_dir("./demo_gemm_quant_int1bxint8b")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    problem_sizes = [
        #(1, 8192, 8192),
        #(1, 28672, 8192),
        #(1, 8192, 28672),
        #(8, 8192, 8192),
        #(8, 28672, 8192),
        #(8, 8192, 28672),
        #(16, 8192, 8192),
        (16, 28672, 8192),
        #(16, 8192, 28672),
        #(32, 8192, 8192),
        #(32, 28672, 8192),
        #(32, 8192, 28672),
    ]

    from tabulate import tabulate

    records = []
    headers = ["problem(m,n,k)", "latency", "idx"]

    scale = 0.01
    for m, n, k in problem_sizes:
        print(f"m, n, k: {m}, {n}, {k}")
        a, b, c = data_intxb(m, n, k, return_hidet=True)
        b_f16 = b.torch().T.contiguous().to(torch.float16)
        bq = torch.empty((k, n // 8), dtype=torch.uint8, device="cuda")

        linear = IntxbAIntxbBGemm(m, k, n, 8, 1, scale=scale, out_dtype=hidet.dtypes.i8)

        best_time = None
        best_module = None
        best_idx = None
        modules = linear.modules()
        for i, quant_module in enumerate(modules):
            module = quant_module.gemm_quant
            qfunc = quant_module.quant.build()
            dqfunc = quant_module.dequant.build()

            bdq = torch.empty((k, n), dtype=torch.float16, device="cuda")
            qfunc(b_f16, bq)
            dqfunc(bq, bdq)

            func = module.build()

            tiled_mma = quant_module._tuning_kwargs["tiled_mma"]
            parallel_k_parts = quant_module._tuning_kwargs["parallel_k_parts"]
            print(quant_module._tuning_kwargs)
            c_shape, _ = tiled_mma.c_tv_layout()
            block_m, block_n = c_shape
            gridm = cdiv(m, block_m)
            gridn = cdiv(n, block_n)

            def fn():
                c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float32, device="cuda")
                c = torch.empty((m, n), dtype=torch.int8, device="cuda")
                counters = torch.empty((gridm, gridn), dtype=torch.int32, device="cuda")
                func(a, bq, c, c_parallel_k_parts, counters)
                return c

            mean, min_lat, max_lat = bench(fn, ())
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            from hidet.utils.benchmark import do_bench

            mean = do_bench(fn, percentiles=None)
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            if best_time is None:
                best_time = mean
                best_module = module
                best_idx = i
            elif mean < best_time:
                best_time = mean
                best_module = module
                best_idx = i

            c = fn()
            c2 = a.torch().to(torch.float16) @ b.torch().to(torch.float16).T
            c2 = (c2 * scale).to(torch.int8)

            import numpy as np
            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), atol=1, rtol=5e-3)

        #a = a.torch()
        #b = b.torch()

        #def fn():
        #    return a @ b.T
        #mean = do_bench(fn, percentiles=None)
        #print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
        #print(
        #    "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
        #        mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
        #    )
        #)
        print(f"m, n, k: {m}, {n}, {k}")
        print(best_time)
        print(best_module)
        print(best_idx)
        records.append([(m, n, k), best_time, best_idx])

    with open("results_quant_gemm_int8bint1b.txt", "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark mixed input GEMM")
    parser.add_argument("--A_bits", type=int, default=8)
    parser.add_argument("--B_bits", type=int, default=8)

    args = parser.parse_args()
    if args.A_bits == 8 and args.B_bits == 8:
        bench_int8bint8b()
    elif args.A_bits == 8 and args.B_bits == 4:
        bench_int8bint4b()
    elif args.A_bits == 8 and args.B_bits == 2:
        bench_int8bint2b()
    elif args.A_bits == 8 and args.B_bits == 1:
        bench_int8bint1b()
    else:
        assert False
