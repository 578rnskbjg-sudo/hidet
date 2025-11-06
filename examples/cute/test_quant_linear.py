import hidet
import torch

from vllm.model_executor.layers.quantization.hidet_kernel import w4a16_linear
from vllm.model_executor.layers.quantization.weight_utils import (
    cast_u4_to_f16,
    cast_f16_to_u4,
    preprocess_weight,
    depreprocess_weight,
    cast_u4_to_f16_interleaved,
)
from vllm._custom_ops import awq_gemm

import time


def data(M, N, K, group_size=64, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    a = torch.randint(low=lo, high=hi, size=(M, K), dtype=dtype, device=device)
    b = torch.randint(low=0, high=hi, size=(K, N), dtype=dtype, device=device)
    scale = torch.randint(low=-1, high=2, size=(K // group_size, N), dtype=dtype, device=device)
    zeros = torch.randint(low=0, high=hi, size=(K // group_size, N), dtype=dtype, device=device)

    ret = [a, b, scale, zeros]

    if return_hidet:
        ret = [hidet.from_torch(x) for x in ret]

    return tuple(ret)


#def bench(f, warmup=1, iter=100):
#    import numpy as np
#
#    cache = torch.empty((int(256e6),), dtype=torch.int8, device='cuda')
#    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(iter)]
#    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(iter)]
#    for i in range(warmup + iter):
#        cache.zero_()
#        if i >= warmup:
#            start_event[i - warmup].record()
#        f()
#        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
#        # happen during realistic model inference as many launches are submitted to the kernel queue.
#        if i >= warmup:
#            end_event[i - warmup].record()
#    torch.cuda.synchronize()
#    times = np.array([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
#    res = np.mean(times).item()
#    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
#    # we execute many benchmarks consecutively
#    time.sleep(1.)
#    return res
def bench2(f, warmup=5, number=1, repeat=31):
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


def bench(f, warmup=10, iter=100):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res * 1000.0


def benchmark_dense(A, B, C):
    def fn():
        torch.matmul(A, B, out=C)

    res = bench2(fn)
    res = res / 1000.0
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10**12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10**9,
    }



def test_quant_linear():
    hidet.option.cache_dir("./demo_quant_linear")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    MODELS = {
        'Llama7B': [(8192, 8192), (28672, 8192), (8192, 28672)],
    }

    #MODELS = {
    #    'Llama7B': [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
    ##    'Llama13B': [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
    ##    'Llama33B': [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
    ##    'Llama65B': [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
    #}

    # Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments
    ALL = False

    for groupsize in [-1, 128] if ALL else [128]:
        print('groupsize=%d' % groupsize)
        print()
        for model, layers in MODELS.items():
            print(model)
            if ALL:
                batchsizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            else:
                batchsizes = [1, 8, 16, 32]
            for batch in batchsizes:
                #if not ALL and model != 'ideal' and batch not in batchsizes:
                #    continue
                tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0}
                for layer in layers:
                    m, n, k = batch, layer[1], layer[0]
                    print(f"m, n, k: {m}, {n}, {k}")
                    a, b, scale, zeros = data(m, n, k, group_size=groupsize, return_hidet=False)
                    bq = cast_f16_to_u4(b)
                    qzeros = cast_f16_to_u4(zeros)
                    zeros1 = cast_u4_to_f16_interleaved(qzeros)
                    linear = w4a16_linear("w4a16", k, n, groupsize)
                    bq4 = preprocess_weight(bq)
                    from hidet.ffi import runtime_api

                    runtime_api.set_symbol_value('m', m)

                    def fn():
                        c = linear(a, bq4, scale, zeros)
                        return c

                    fn()
                    mean = bench2(fn, repeat=100)

                    print("==========================")
                    print("{}x{}x{}".format(m, n, k))
                    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
                    print(
                        "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                            mean, (m * n * 2 + m * k * 2 + k * n // 2) / (1e6 * mean)
                        )
                    )

                    c = linear(a, bq4, scale, zeros1)
                    bqq = cast_u4_to_f16_interleaved(bq)
                    b2 = bqq.view(k // groupsize, groupsize, n)
                    b3 = scale.view(k // groupsize, 1, n) * (b2 - zeros1.view(k // groupsize, 1, n))
                    b = b3.view(k, n)
                    c2 = a @ b
                    import numpy as np

                    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
                    # np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=5e-3)

                    split_k_factor = 8
                    c3 = awq_gemm(a, bq, scale, qzeros, split_k_factor)
                    # np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c3.cpu().numpy(), rtol=5e-3)

                    res_q = {}
                    res_q['s'] = mean / 1000.0
                    res_q['TFLOP/s'] = 2.0 * (m * n * k) / (1e9 * mean)
                    res_q['GB/s'] = (m * n * 2 + m * k * 2 + k * n // 2) / (1e6 * mean)

                    res_d = benchmark_dense(a, b, c)
                    res_q['speedup'] = res_d['s'] / res_q['s']
                    tot_q['s'] += res_q['s']
                    for k in tot_q:
                        if k != 's':
                            tot_q[k] += res_q[k] * res_q['s']
                for k in tot_q:
                    if k != 's':
                        tot_q[k] /= tot_q['s']
                print(
                    'batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f'
                    % (batch, tot_q['s'], tot_q['TFLOP/s'], tot_q['GB/s'], tot_q['speedup'])
                )
            print()
