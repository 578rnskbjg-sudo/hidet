#!/bin/bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1410
source $root/workspace/vllm_v8_2/bin/activate
python3 $root/workspace/hidet/examples/cute/benchmark_mamba_scan.py --cache-dir ./demo_mamba_scan --output ./mamba_scan.txt
deactivate
sudo nvidia-smi -rgc