import sys

import numpy as np
import torch
import marlin

import time


def benchmark(f, warmup=10, iter=100):
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
    return res

#def benchmark(f, warmup=1, iter=100):
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
#    return res * 1e-3


def bench2(f, warmup=5, number=1, repeat=31):
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


def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s


def benchmark_dense(A, B, C):
    res = bench2(lambda: torch.matmul(A, B, out=C), repeat=100)
    res = res / 1000
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }


def benchmark_quant(A, B, C, s, thread_k, thread_n, sms):
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device('cuda:0'))
    res = bench2(lambda: marlin.mul(A, B, C, s, workspace, thread_k, thread_n, sms), repeat=100)
    res = res / 1000
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
    }

# Pass the SM count for known GPUs to avoid the kernel having to query this information (this is very minor)
gpu = torch.cuda.get_device_name(0)
if 'A100' in gpu:
    SMS = 108
elif 'A10' in gpu:
    SMS = 72
elif '3090' in gpu:
    SMS = 82
elif 'A6000' in gpu:
    SMS = 84
else:
    SMS = -1

MODELS = {
    'Llama7B': [(8192, 8192), (28672, 8192), (8192, 28672)],
}

#MODELS = {
#    #'ideal': [
#    #    (4 * 256 * SMS, 256 * SMS)
#    #],
#    'Llama7B': [
#        (4096, 3 * 4096),
#        (4096, 4096),
#        (4096, 2 * 10752),
#        (10752, 4096)
#    ],
#    'Llama13B': [
#        (5120, 3 * 5120),
#        (5120, 5120),
#        (5120, 2 * 13568),
#        (13568, 5120)
#    ],
#    'Llama33B': [
#        (6656, 3 * 6656),
#        (6656, 6656),
#        (6656, 2 * 17664),
#        (17664, 6656)
#    ],
#    'Llama65B': [
#        (8192, 3 * 8192),
#        (8192, 8192),
#        (8192, 2 * 21760),
#        (21760, 8192)
#    ],
#    #'Falcon180B': [
#    #    # Note that parallel attention and FC allows layer fusions
#    #    (14848, 14848 * 5 + 1024),
#    #    (14848 * 5, 14848)
#    #]
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
            #if not ALL and model != 'ideal' and batch not in [1]:
            #    continue
            tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0}
            for layer in layers:
                A, B, C, B_ref, s = get_problem(batch, layer[1], layer[0], groupsize)
                res_d = benchmark_dense(A, B_ref, C)
                if model == 'ideal' and batch == 16:
                    # This is a special case constructed to be optimal for a thread-shape different than the default one
                    res_q = benchmark_quant(A, B, C, s, 64, 256, SMS)
                else:
                    res_q = benchmark_quant(A, B, C, s, -1, -1, SMS)
                mean = res_q['s'] * 1000
                m, n, k = batch, layer[1], layer[0]
                print("==========================")
                print("{}x{}x{}".format(m, n, k))
                print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
                print(
                    "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                        mean, (m * n * 2 + m * k * 2 + k * n // 2) / (1e6 * mean)
                    )
                )
                print("Dense")
                mean = res_d['s'] * 1000
                print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
                print(
                    "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                        mean, (m * n * 2 + m * k * 2 + k * n // 2) / (1e6 * mean)
                    )
                )
                res_q['speedup'] = res_d['s'] / res_q['s']
                tot_q['s'] += res_q['s']
                for k in tot_q:
                    if k != 's':
                        tot_q[k] += res_q[k] * res_q['s']
            for k in tot_q:
                if k != 's':
                    tot_q[k] /= tot_q['s']
            print('batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f' % (
                batch,
                tot_q['s'],
                tot_q['TFLOP/s'],
                tot_q['GB/s'],
                tot_q['speedup']
            ))
        print()
