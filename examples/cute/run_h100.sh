# run on hopper
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1410
source $root/workspace/vllm_v8_2/bin/activate
python3 $root/workspace/hidet/examples/cute/benchmark_warp_specialized_gemm_non_persistent.py --m 4096 --n 4096 --k 4096 --search-space 2 --cache-dir ./demo_hopper_gemm --output ./warp_specialized_gemm_expt.txt
python3 $root/workspace/hidet/examples/cute/benchmark_scaled_mm.static.py --m 4096 --n 4096 --k 4096 --group-n 128 --group-k 128 --cache-dir ./demo_hopper_gemm_f8 --output w8a8_scaled_mm.txt
deactivate
# flash attention needs to run with torch 2.7
source $root/workspace/vllm_v9_2/bin/activate
python3 $root/workspace/hidet/examples/cute/benchmark_flash_attention3.py --batch-size 1 --num-heads 32 --num-heads-k 32 --head-size 128 --seqlen-q 16384 --seqlen-k 16384 --cache-dir attn_flash3 --output attention_flash3.txt
deactivate
sudo nvidia-smi -rgc