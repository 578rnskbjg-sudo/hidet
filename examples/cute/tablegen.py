from re import I
import numpy as np

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


def compute_geomean(triton, cuda, hexcute):
    triton = [float(x) for x in triton]
    cuda = [float(x) for x in cuda]
    hexcute = [float(x) for x in hexcute]
    nr_data = len(triton)
    assert nr_data == len(cuda) == len(hexcute)
    arr_triton = np.array(triton)
    arr_cuda = np.array(cuda)
    arr_hexcute = np.array(hexcute)
    geomean_triton = np.prod(arr_triton/arr_cuda) ** (1/nr_data)
    geomean_hexcute = np.prod(arr_hexcute/arr_cuda) ** (1/nr_data)
    return geomean_triton, geomean_hexcute


def main():
    _, _, _, _, flops_triton_ws_gemm, flops_cublas_ws_gemm, flops_hexcute_ws_gemm = markdown_table_to_dicts("warp_specialized_gemm_expt.txt")
    _, _, _, _, flops_triton_scaled_mm, flops_cutlass_scaled_mm, flops_hexcute_scaled_mm = markdown_table_to_dicts("w8a8_scaled_mm.txt")
    _, _, _, _, _, _, hexcute_flash3, triton_flash3, flashattn_3 = markdown_table_to_dicts("attention_flash3.txt")
    _, _, _, _, flops_cublas_gemm, flops_triton_gemm, flops_hexcute_gemm = markdown_table_to_dicts("matmul_a100.txt")
    _, _, _, _, _, _, _, _, _, _, bw_hexcute, bw_flash_attn, bw_flash_infer, bw_triton = markdown_table_to_dicts("decoding_a100.txt")
    _, _, _, _, _, _, hexcute_attn, triton_attn, flashattn_attn = markdown_table_to_dicts("attn_forward.txt")

    hexcute_flash3 = [1/float(x) for x in hexcute_flash3]
    triton_flash3 = [1/float(x) for x in triton_flash3]
    flashattn_3 = [1/float(x) for x in flashattn_3]
    hexcute_attn = [1/float(x) for x in hexcute_attn]
    triton_attn = [1/float(x) for x in triton_attn]
    flashattn_attn = [1/float(x) for x in flashattn_attn]
    
    triton_ws_gemm, hexcute_ws_gemm = compute_geomean(flops_triton_ws_gemm, flops_cublas_ws_gemm, flops_hexcute_ws_gemm)
    triton_scaled_mm, hexcute_scaled_mm = compute_geomean(flops_triton_scaled_mm, flops_cutlass_scaled_mm, flops_hexcute_scaled_mm)
    triton_flash3, hexcute_flash3 = compute_geomean(triton_flash3, flashattn_3, hexcute_flash3)
    triton_gemm, hexcute_gemm = compute_geomean(flops_triton_gemm, flops_cublas_gemm, flops_hexcute_gemm)
    triton_attn, hexcute_attn = compute_geomean(triton_attn, flashattn_attn, hexcute_attn)
    triton_decoding, hexcute_decoding = compute_geomean(bw_triton, bw_flash_infer, bw_hexcute)
    
    from tabulate import tabulate
    headers = ["Operator", "Evaluated Shapes", "CUDA LoC", "Triton LoC", "Hexcute LoC", "Performance Baseline", "Triton (Normalized Perf)", "Hexcute (Normalized Perf)"]   
    records = []
    records.append(["Blockwise Scaled FP8 GEMM", len(flops_cutlass_scaled_mm), 900, 87, 180, "CUTLASS", triton_scaled_mm, hexcute_scaled_mm])
    records.append(["Warp Specialized FP16 GEMM", len(flops_cublas_ws_gemm), 1024, 71, 169, "cuBLAS", triton_ws_gemm, hexcute_ws_gemm])
    records.append(["Fused MHA Forward", len(flashattn_3), 1684, 114, 212, "FlashAttention3", triton_flash3, hexcute_flash3])
    print("NVIDIA H100 GPU") 
    print(tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left"))

    records = []
    records.append(["FP16 GEMM", len(flops_cublas_gemm), 703, 71, 98, "cuBLAS", triton_gemm, hexcute_gemm])
    records.append(["Fused MHA Forward", len(flashattn_attn), 577, 114, 172, "FlashAttention", triton_attn, hexcute_attn])
    records.append(["Fused MHA Decoding", len(bw_flash_infer), 322, 224, 253, "FlashInfer", triton_decoding, hexcute_decoding])
    print("NVIDIA A100 GPU") 
    print(tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left"))

if __name__ == "__main__":
    main()