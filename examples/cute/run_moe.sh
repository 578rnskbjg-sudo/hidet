# run on hopper
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 2000
# run moe
source $root/workspace/vllm_v9_2/bin/activate
python3 $root/workspace/hidet/examples/cute/test_moe.py
deactivate
# run moe v8.2
source $root/workspace/vllm_v8_2/bin/activate
python3 $root/workspace/hidet/examples/cute/test_moe_v8.2.py
deactivate
sudo nvidia-smi -rgc