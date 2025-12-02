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
    geomean_triton = np.prod(arr_triton / arr_cuda) ** (1 / nr_data)
    geomean_hexcute = np.prod(arr_hexcute / arr_cuda) ** (1 / nr_data)
    return geomean_triton, geomean_hexcute


def main():
    import os
    if os.path.exists("warp_specialized_gemm_expt.txt"):
        _, _, _, _, flops_triton_ws_gemm, flops_cublas_ws_gemm, flops_hexcute_ws_gemm = markdown_table_to_dicts("warp_specialized_gemm_expt.txt")
        triton_ws_gemm, hexcute_ws_gemm = compute_geomean(flops_triton_ws_gemm, flops_cublas_ws_gemm, flops_hexcute_ws_gemm)
    else:
        triton_ws_gemm = 0
        hexcute_ws_gemm = 0
    
    if os.path.exists("w8a8_scaled_mm.txt"):
        _, _, _, _, flops_triton_scaled_mm, flops_cutlass_scaled_mm, flops_hexcute_scaled_mm = markdown_table_to_dicts("w8a8_scaled_mm.txt")
        triton_scaled_mm, hexcute_scaled_mm = compute_geomean(flops_triton_scaled_mm, flops_cutlass_scaled_mm, flops_hexcute_scaled_mm)
    else:
        triton_scaled_mm = 0
        hexcute_scaled_mm = 0
    
    if os.path.exists("attention_flash3.txt"):
        _, _, _, _, _, _, hexcute_flash3, triton_flash3, flashattn_3 = markdown_table_to_dicts("attention_flash3.txt")
        hexcute_flash3 = [1 / float(x) for x in hexcute_flash3]
        triton_flash3 = [1 / float(x) for x in triton_flash3]
        flashattn_3 = [1 / float(x) for x in flashattn_3]
        triton_flash3, hexcute_flash3 = compute_geomean(triton_flash3, flashattn_3, hexcute_flash3)
    else:
        triton_flash3 = 0
        hexcute_flash3 = 0
    
    if os.path.exists("matmul_a100.txt"):
        _, _, _, _, flops_cublas_gemm, flops_triton_gemm, flops_hexcute_gemm = markdown_table_to_dicts("matmul_a100.txt")
        triton_gemm, hexcute_gemm = compute_geomean(flops_triton_gemm, flops_cublas_gemm, flops_hexcute_gemm)
    else:
        triton_gemm = 0
        hexcute_gemm = 0
    
    if os.path.exists("decoding_a100.txt"):
        _, _, _, _, _, _, bw_hexcute, bw_flash_attn, bw_flash_infer, bw_triton = markdown_table_to_dicts("decoding_a100.txt")
        triton_decoding, hexcute_decoding = compute_geomean(bw_triton, bw_flash_infer, bw_hexcute)
    else:
        triton_decoding = 0
        hexcute_decoding = 0
    
    if os.path.exists("attn_forward.txt"):
        _, _, _, _, _, _, hexcute_attn, triton_attn, flashattn_attn = markdown_table_to_dicts("attn_forward.txt")
        hexcute_attn = [1 / float(x) for x in hexcute_attn]
        triton_attn = [1 / float(x) for x in triton_attn]
        flashattn_attn = [1 / float(x) for x in flashattn_attn]
        triton_attn, hexcute_attn = compute_geomean(triton_attn, flashattn_attn, hexcute_attn)
    else:
        triton_attn = 0
        hexcute_attn = 0


    table_template = r"""
\documentclass[10pt,a4paper]{article}
\usepackage{tabularx}
\usepackage{multirow, makecell}
\usepackage[bottom=0.5cm, right=0.5cm, left=0.5cm, top=1.5cm]{geometry}

\begin{document}
\begin{table*}[t!]
{\scriptsize
\begin{center}
\begin{tabular}{cccccccccc}
\hline
 \multirow{2}{*}{\textbf{GPU}} & \multirow{2}{*}{\textbf{Operator}} & \multirowcell{2}{\textbf{Evaluated}\\\textbf{Shapes}} & \multicolumn{3}{c}{\textbf{Lines of Code}}&\multirow{2}{*}{\textbf{Performance Baseline}}&\multicolumn{2}{c}{\textbf{Normalized Performance}} \\
 & & & \textbf{CUDA}& \textbf{Triton}& \textbf{Hexcute}& &\textbf{Triton}& \textbf{Hexcute}
\\
\hline
\multirowcell{3}{NVIDIA\\ A100 GPU} & FP16 GEMM&40& 703$^\mathrm{b}$ & 71 & 98 & cuBLAS & TRITON_GEMM$\times$ & \textbf{HEXCUTE_GEMM$\times$} &  \\
& Fused MHA$^\mathrm{a}$ Forward&20& 577 & 114 & 172 & FlashAttention2 & TRITON_ATTN$\times$ & \textbf{HEXCUTE_ATTN$\times$} \\
& Fused MHA Decoding&24& 322 & 224 & 253 & FlashInfer & TRITON_DECODING$\times$ & \textbf{HEXCUTE_DECODING$\times$} \\
\hline
\multirowcell{3}{NVIDIA \\ H100 GPU} & Blockwise Scaled FP8 GEMM & 35 & 900 & 87 & 180 & CUTLASS & TRITON_SCALED_MM$\times$ & \textbf{HEXCUTE_SCALED_MM$\times$} \\
%\hline
& Warp Specialized FP16 GEMM & 40 & 1024$^\mathrm{b}$ & 71 & 169 & cuBLAS & TRITON_WS_GEMM$\times$ & \textbf{HEXCUTE_WS_GEMM$\times$} \\
%\hline
& Fused MHA Forward & 20 & 1684 & 114 & 212 & FlashAttention3 & TRITON_FLASH3$\times$ & \textbf{HEXCUTE_FLASH3$\times$} \\
%& Mixed-type Mixture of Expert &  &  & & & Marlin & & \\
\hline
\multicolumn{10}{l}{$^{\mathrm{a}}$ Multi-head Attention}\\
\multicolumn{10}{l}{$^{\mathrm{b}}$ For GEMM, speedups are reported against cuBLAS, while LoC comparisons use CUTLASS because cuBLAS is closed‑source.}
\end{tabular}
\label{general-operators}
\end{center}
}
\end{table*}

\end{document}
"""
    with open("Table_II.tex", "w") as f:
        f.write(
            table_template.replace("TRITON_GEMM", f"{triton_gemm:.2f}" if triton_gemm > 0 else "")
            .replace("HEXCUTE_GEMM", f"{hexcute_gemm:.2f}" if hexcute_gemm > 0 else "")
            .replace("TRITON_ATTN", f"{triton_attn:.2f}" if triton_attn > 0 else "")
            .replace("HEXCUTE_ATTN", f"{hexcute_attn:.2f}" if hexcute_attn > 0 else "")
            .replace("TRITON_DECODING", f"{triton_decoding:.2f}" if triton_decoding > 0 else "")
            .replace("HEXCUTE_DECODING", f"{hexcute_decoding:.2f}" if hexcute_decoding > 0 else "")
            .replace("TRITON_SCALED_MM", f"{triton_scaled_mm:.2f}" if triton_scaled_mm > 0 else "")
            .replace("HEXCUTE_SCALED_MM", f"{hexcute_scaled_mm:.2f}" if hexcute_scaled_mm > 0 else "")
            .replace("TRITON_WS_GEMM", f"{triton_ws_gemm:.2f}" if triton_ws_gemm > 0 else "")
            .replace("HEXCUTE_WS_GEMM", f"{hexcute_ws_gemm:.2f}" if hexcute_ws_gemm > 0 else "")
            .replace("TRITON_FLASH3", f"{triton_flash3:.2f}" if triton_flash3 > 0 else "")
            .replace("HEXCUTE_FLASH3", f"{hexcute_flash3:.2f}" if hexcute_flash3 > 0 else "")
        )

if __name__ == "__main__":
    main()
