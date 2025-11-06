import hidet
import torch


class Kernel:
    def build(self):
        from hashlib import sha256
        from hidet.drivers import build_ir_module
        from hidet.runtime import load_compiled_module
        from tqdm import tqdm
        from hidet.utils.multiprocess import parallel_imap_2ndlevel

        def build_job(args):
            ir_module, output_dir = args
            build_ir_module(ir_module, output_dir, target='cuda')

        ir_modules = self.modules()
        output_dirs = []
        for mod in ir_modules:
            hash_dir = sha256(str(mod).encode()).hexdigest()[:16]
            output_dir = hidet.utils.cache_dir('ir_modules', hash_dir)
            output_dirs.append(output_dir)

        jobs = [(ir_module, output_dir) for ir_module, output_dir in zip(ir_modules, output_dirs)]

        for _ in tqdm(
            parallel_imap_2ndlevel(build_job, jobs, is_remote_allowed=True), desc="Compiling", total=len(jobs), ncols=80
        ):
            pass
        funcs = [load_compiled_module(output_dir) for output_dir in output_dirs]
        return list(zip(ir_modules, funcs))


class GDNAttention(Kernel):
    def __init__(self, H, Hg, K, chunk_size: int = 64, IS_VARLEN: bool = True):
        self.H = H
        self.Hg = Hg
        assert K <= 256
        self.K = K
        self.chunk_size = chunk_size
        assert chunk_size == 64
        self.IS_VARLEN = IS_VARLEN

    def modules(self):
        return [self.extract_ir_module(self._experimental)]

    def _experimental(self, tiled_mmas: TiledMma, BK: int, stages: int):
        from hidet.lang.cute import for_each_thread, for_each_warp
        import hidet.lang.cute as ct
        from hidet.ir.cute import TensorLayout
        from hidet.ir.cute.ops import tensor_view, make_tensor, cast, mma, fill, transpose
        from hidet.ir.dtypes import f16, f32
        from hidet.lang import attrs, tensor_pointer, grid
        from hidet.ir.layout import row_major
        from hidet.lang.cuda import dynamic_shared_memory, syncthreads
        from hidet.ir.primitives.cuda import syncwarp, cp_async_commit_group, cp_async_wait_all
        from hidet.ir.cute import MmaAtom, TiledMma, Level

        tiled_mma_A = tiled_mmas[0]
        k_shape, _ = tiled_mma_A.a_tv_layout()
        BT, TileK = k_shape
        tune.check(BT == self.chunk_size)
        
        H = self.H
        Hg = self.Hg
        K = self.K

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                k: f16[tokens, Hg, K],
                beta: beta_dtype[tokens, H],
                g_cumsum: float32[tokens, H],
                cu_seqlens: i32[reqs + 1],
                chunk_indices: i32[2 * blocks],
                A: f16[tokens, H, BT],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = 128, 1, 1
                attrs.cuda.grid_dim = blocks, 1, 1
                attrs.cuda.dynamic_smem_bytes = 64 * 64 * f16.nbytes

                i_bt = blockIdx.x
                i_h = blockIdx.y
                if IS_VARLEN:
                    i_b, i_t = chunk_indices[i_bt * 2], chunk_indices[i_bt * 2 + 1]
                    bos = cu_seqlens[i_b]
                    eos = cu_seqlens[i_b + 1]
                    T = eos - bos
                else:
                    i_b = i_bt // T
                    i_t = i_bt % T
                    bos = i_t * T
                    eos = i_t * T + T

                gK = ct.tma_tensor(
                    k,
                    shape=(tokens, Hg * K),
                    stride=(Hg * K, 1),
                    tile_shape=(BT, BK),
                    tile_coords=(bos + i_t * BT, i_h // (H // Hg) * K),
                )
                gBeta = ct.global_view(beta + bos * H + i_t * BT * H + i_h, shape=(BT, BK), stride=(H, 0))
                gGcumsum_row = ct.global_view(g_cumsum + bos * H + i_t * BT * H + i_h, shape=(BT, BT), stride=(H, 0))
                gGcumsum_col = ct.global_view(g_cumsum + bos * H + i_t * BT * H + i_h, shape=(BT, BT), stride=(0, H))

                sK = ct.shared_tensor(f16, (BT, K))
                sKb = ct.shared_tensor(f16, (BT, BK, 2))
                tAsK = partition_A(sKb, tiled_mma_A)
                tBsK = partition_B(sKb, tiled_mma_A)

                tXgK = ct.partition_S(gK, (BT, BK))
                tXgBeta = ct.partition_S(gBeta, (BT, BK))
                rK = ct.register_tensor(f16, (BT, BK))
                tXrK = ct.partition_D(rK, (BT, BK))
                tXrBeta = ct.partition_D(gBeta, (BT, BK))

                tKsK = ct.partition_D(sK, (BT, BK))
                tKsKb = ct.partition_D(sKb, (BT, BK))
                rA = ct.register_tensor(f32, (BT, BT))
                fill(rA, 0.0)

                ct.tile_copy((BT, BK), tXgK[:, :, 0], tXrK)
                ct.tile_copy((BT, BK), tXgBeta, tXrBeta)
                kb = cast(cast(tXrK, f32) * cast(tXrBeta, f32), f16)
                tKrKb = ct.partition_S(kb, (BT, BK))
                ct.tile_copy((BT, BK), tKrKb, tKsKb[:, :, 0])
                ct.tile_copy((BT, BK), tKrK, tKsK[:, :, 0])
                syncthreads()

                for tile_k in grid(cdiv(BK, TileK), attrs='u+'):
                    mma(tiled_mma_A, rA, tAsK[:, :, tile_k, 0], tBsK[:, :, tile_k, 0], rA)

                ct.tile_copy((BT, BK), tXgK[:, :, 1], tXrK)
                kb = cast(cast(tXrK, f32) * cast(tXrBeta, f32), f16)
                tKrKb = ct.partition_S(kb, (BT, BK))
                ct.tile_copy((BT, BK), tKrKb, tKsKb[:, :, 1])
                ct.tile_copy((BT, BK), tKrK, tKsK[:, :, 1])
                syncthreads()

                BlockK = cdiv(K, BK)
                for i_k in range(BlockK - 1):
                    if i_k < BlockK - 2:
                        ct.tile_copy((BT, BK), tXgK[:, :, i_k + 2], tXrK)

                    for tile_k in grid(cdiv(BK, TileK), attrs='u+'):
                        mma(tiled_mma_A, rA, tAsK[:, :, tile_k, (i_k + 1) % 2], tBsK[:, :, tile_k, (i_k + 1) % 2], rA)

                    if i_k < BlockK - 2:
                        kb = cast(cast(tXrK, f32) * cast(tXrBeta, f32), f16)
                        tKrKb = ct.partition_S(kb, (BT, BK))
                        ct.tile_copy((BT, BK), tKrK, tKsK[:, :, i_k + 2])
                        wgmma_wait_group(1)
                        ct.tile_copy((BT, BK), tKrKb, tKsKb[:, :, i_k % 2])

                    syncthreads()

                wgmma_wait_group(0)
                rGcumsum_row = ct.register_tensor(f32, (BT, BT), stride_hint=(1, 0))
                rGcumsum_col = ct.register_tensor(f32, (BT, BT), stride_hint=(0, 1))
                ct.tile_copy((BT, BT), gGcumsum_row, rGcumsum_row)
                ct.tile_copy((BT, BT), gGcumsum_col, rGcumsum_col)
                rGdiff = rGcumsum_row - rGcumsum_col
                rAg = rA * exp(rGdiff)
                # TODO: add a tril operator in Hexcute
                sA = ct.tensor_pointer('float16', shape=[BT, BT], layout=row_major(BT, BT))
                sA = dynamic_shared_memory(0, f16)
                tArA = ct.partition_S(cast(rAg, f32), (BT, BT))
                tAsA = ct.partition_D(sA, (BT, BT))
                ct.tile_copy((BT, BT), tArA, tAsA)
                syncthreads()
                inverse_64x64_tile(sA)

                gA = ct.tma_tensor(
                    A,
                    shape=(tokens, H * BT),
                    stride=(H * BT, 1),
                    tile_shape=(BT, BT),
                    tile_coords=(bos + i_t * BT, i_h * BT),
                )
                ct.tile_copy(sA, gA)

            rmem_layout = ct.make_register_layout("thread", (1, 1), TensorLayout((1, 1), (1, 1)))
            a = TensorLayout(((1,), (2, 2)), ((1,), (2, 1)))
            b = TensorLayout(((1,), (2, 2)), ((1,), (1, 2)))
            c = TensorLayout(((1,), (2, 2)), ((1,), (2, 1)))
            mma_atom = MmaAtom("thread", (2, 2, 2), a, b, c, c)
            tiled_mma_2x2 = TiledMma(mma_atom, [])
            mma_atom = MmaAtom("thread", (2, 2, 2), a, b, c, c, (2, 2))
            tiled_mma_4x4 = TiledMma(mma_atom, [])

            a = TensorLayout(((1,), (1, 2)), ((1,), (1, 1)))
            b = TensorLayout(((1,), (2, 2)), ((1,), (1, 2)))
            c = TensorLayout(((1,), (1, 2)), ((1,), (1, 1)))
            mma_atom = MmaAtom("thread", (1, 2, 2), a, b, c, c, (1, 1))
            thread_in_warp = Level("thread", "warp", (8, 4), TensorLayout((8, 4), (1, 8)))
            tiled_mma_8x8 = TiledMma(mma_atom, [thread_in_warp])

            a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
            b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
            c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
            mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
            tiled_mma_16x16 = TiledMma(mma_atom, [])

            warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (1, 1))
            tiled_mma_32x32 = TiledMma(mma_atom, [warp_in_threadblock])

            @hidet.script
            def inverse_64x64_tile(sA: f16[64, 64]):
                with for_each_thread((0, 64)) as i:
                    sA_11 = tensor_view(sA[i : i + 1, i : i + 1], TensorLayout((1, 1), (64, 1)), "shared")
                    rA_11 = make_tensor(f16, rmem_layout, "register")
                    ct.copy(sA_11, rA_11)
                    rA_11 = cast(f32(1.0) / cast(rA_11, f32), f16)
                    ct.copy(rA_11, sA_11)
                syncthreads()

                with for_each_thread((0, 32)) as i:
                    sA_11 = tensor_view(
                        sA[2 * i : 2 * i + 1, 2 * i : 2 * i + 1], TensorLayout((1, 1), (64, 1)), "shared"
                    )
                    rA_11 = make_tensor(f16, rmem_layout, "register")
                    ct.copy(sA_11, rA_11)
                    sA_21 = tensor_view(
                        sA[2 * i + 1 : 2 * i + 2, 2 * i : 2 * i + 1], TensorLayout((1, 1), (64, 1)), "shared"
                    )
                    rA_21 = ct.register_tensor(f16, (1, 1))
                    ct.copy(sA_21, rA_21)
                    sA_22 = tensor_view(
                        sA[2 * i + 1 : 2 * i + 2, 2 * i + 1 : 2 * i + 2], TensorLayout((1, 1), (64, 1)), "shared"
                    )
                    rA_22 = ct.register_tensor(f16, (1, 1))
                    ct.copy(sA_22, rA_22)
                    rA_21_inv = cast(f32(-1.0) * cast(rA_22, f32) * cast(rA_21, f32) * cast(rA_11, f32), f16)
                    tXrA_21_inv = ct.partition_S(rA_21_inv, (1, 1))
                    tXsA_21 = ct.partition_D(sA_21, (1, 1))
                    ct.tile_copy((1, 1), tXrA_21_inv, tXsA_21)
                syncwarp()

                with for_each_thread((0, 16)) as i:
                    sA_11 = tensor_view(
                        sA[4 * i : 4 * i + 2, 4 * i : 4 * i + 2], TensorLayout((2, 2), (1, 64)), "shared"
                    )
                    sA_21 = tensor_view(
                        sA[4 * i + 2 : 4 * i + 4, 4 * i : 4 * i + 2], TensorLayout((2, 2), (64, 1)), "shared"
                    )
                    sA_22 = tensor_view(
                        sA[4 * i + 2 : 4 * i + 4, 4 * i + 2 : 4 * i + 4], TensorLayout((2, 2), (64, 1)), "shared"
                    )
                    rA_11 = ct.register_tensor(f16, (2, 2))
                    rA_21 = ct.register_tensor(f16, (2, 2))
                    rA_2111 = ct.register_tensor(f16, (2, 2))
                    fill(rA_2111, f16(0.0))
                    ct.copy(sA_21, rA_21)
                    ct.copy(sA_11, rA_11)
                    mma(tiled_mma_2x2, rA_2111, rA_21, rA_11, rA_2111)
                    ct.copy(rA_2111, sA_21)
                    sA_21_t = transpose(sA_21, 1, 0)
                    rA_21_t = ct.register_tensor(f16, (2, 2))
                    rA_22 = ct.register_tensor(f16, (2, 2))
                    rA_2221 = ct.register_tensor(f16, (2, 2))
                    fill(rA_2221, f16(0.0))
                    ct.copy(sA_22, rA_22)
                    ct.copy(sA_21_t, rA_21_t)
                    mma(tiled_mma_2x2, rA_2221, rA_22, rA_21_t, rA_2221)
                    rA_21_inv = f16(-1.0) * rA_2221
                    tXrA_21_inv = ct.partition_S(rA_21_inv, (2, 2))
                    tXsA_21 = ct.partition_D(sA_21, (2, 2))
                    ct.tile_copy((2, 2), tXrA_21_inv, tXsA_21)
                syncwarp()

                with for_each_thread((0, 8)) as i:
                    sA_11 = tensor_view(
                        sA[8 * i : 8 * i + 4, 8 * i : 8 * i + 4], TensorLayout((4, 4), (1, 64)), "shared"
                    )
                    sA_21 = tensor_view(
                        sA[8 * i + 4 : 8 * i + 8, 8 * i : 8 * i + 4], TensorLayout((4, 4), (64, 1)), "shared"
                    )
                    sA_22 = tensor_view(
                        sA[8 * i + 4 : 8 * i + 8, 8 * i + 4 : 8 * i + 8], TensorLayout((4, 4), (64, 1)), "shared"
                    )
                    rA_11 = ct.register_tensor(f16, (4, 4))
                    rA_21 = ct.register_tensor(f16, (4, 4))
                    rA_2111 = ct.register_tensor(f16, (4, 4))
                    fill(rA_2111, f16(0.0))
                    tXrA_11 = ct.partition_D(rA_11, (4, 2))
                    tXrA_21 = ct.partition_D(rA_21, (4, 2))
                    tXsA_11 = ct.partition_S(sA_11, (4, 2))
                    tXsA_21 = ct.partition_S(sA_21, (4, 2))
                    ct.tile_copy((4, 2), tXsA_11[:, :, 0], tXrA_11[:, :, 0])
                    ct.tile_copy((4, 2), tXsA_21[:, :, 0], tXrA_21[:, :, 0])
                    ct.tile_copy((4, 2), tXsA_11[:, :, 1], tXrA_11[:, :, 1])
                    ct.tile_copy((4, 2), tXsA_21[:, :, 1], tXrA_21[:, :, 1])
                    mma(tiled_mma_4x4, rA_2111, tXrA_21[:, :, 0], tXrA_11[:, :, 0], rA_2111)
                    mma(tiled_mma_4x4, rA_2111, tXrA_21[:, :, 1], tXrA_11[:, :, 1], rA_2111)
                    sA_21_t = transpose(sA_21, 1, 0)
                    rA_21_t = ct.register_tensor(f16, (4, 4))
                    rA_22 = ct.register_tensor(f16, (4, 4))
                    rA_2221 = ct.register_tensor(f16, (4, 4))
                    fill(rA_2221, f16(0.0))
                    tXrA_21_t = ct.partition_D(rA_21_t, (4, 2))
                    tXsA_21_t = ct.partition_S(sA_21_t, (4, 2))
                    tXrA_22 = ct.partition_D(rA_22, (4, 2))
                    tXsA_22 = ct.partition_S(sA_22, (4, 2))
                    ct.tile_copy((4, 2), tXsA_21_t[:, :, 0], tXrA_21_t[:, :, 0])
                    ct.tile_copy((4, 2), tXsA_22[:, :, 0], tXrA_22[:, :, 0])
                    ct.tile_copy((4, 2), tXsA_21_t[:, :, 1], tXrA_21_t[:, :, 1])
                    ct.tile_copy((4, 2), tXsA_22[:, :, 1], tXrA_22[:, :, 1])
                    mma(tiled_mma_4x4, rA_2221, tXrA_22[:, :, 0], tXrA_21_t[:, :, 0], rA_2221)
                    mma(tiled_mma_4x4, rA_2221, tXrA_22[:, :, 1], tXrA_21_t[:, :, 1], rA_2221)
                    rA_21_inv = f16(-1.0) * rA_2221
                    tXrA_21_inv = ct.partition_S(rA_21_inv, (4, 4))
                    tXsA_21_inv = ct.partition_D(sA_21, (4, 4))
                    ct.tile_copy((4, 4), tXrA_21_inv, tXsA_21_inv)
                syncthreads()

                with for_each_warp((0, 4)) as i:
                    sA_11 = tensor_view(
                        sA[16 * i : 16 * i + 8, 16 * i : 16 * i + 8],
                        TensorLayout((8, 8), (1, 64)),
                        "shared",
                        volatile=True,
                    )
                    sA_21 = tensor_view(
                        sA[16 * i + 8 : 16 * i + 16, 16 * i : 16 * i + 8],
                        TensorLayout((8, 8), (64, 1)),
                        "shared",
                        volatile=True,
                    )
                    rA_11 = ct.register_tensor(f16, (8, 4))
                    rA_21 = ct.register_tensor(f16, (8, 4))
                    rA_2111 = ct.register_tensor(f16, (8, 8))
                    fill(rA_2111, f16(0.0))
                    tXrA_11 = ct.partition_D(rA_11, (8, 2))
                    tXrA_21 = ct.partition_D(rA_21, (8, 2))
                    tXsA_11 = ct.partition_S(sA_11, (8, 2))
                    tXsA_21 = ct.partition_S(sA_21, (8, 2))
                    ct.tile_copy((8, 2), tXsA_11[:, :, 0], tXrA_11[:, :, 0])
                    ct.tile_copy((8, 2), tXsA_21[:, :, 0], tXrA_21[:, :, 0])
                    for k in grid(4, attrs="u+"):
                        mma(tiled_mma_8x8, rA_2111, tXrA_21[:, :, k % 2], tXrA_11[:, :, k % 2], rA_2111)
                        if k < 3:
                            ct.tile_copy((8, 2), tXsA_11[:, :, k + 1], tXrA_11[:, :, (k + 1) % 2])
                            ct.tile_copy((8, 2), tXsA_21[:, :, k + 1], tXrA_21[:, :, (k + 1) % 2])
                    ct.copy(rA_2111, sA_21)
                    sA_21_t = transpose(sA_21, 1, 0)
                    sA_22 = tensor_view(
                        sA[16 * i + 8 : 16 * i + 16, 16 * i + 8 : 16 * i + 16],
                        TensorLayout((8, 8), (64, 1)),
                        "shared",
                        volatile=True,
                    )
                    rA_21_t = ct.register_tensor(f16, (8, 4))
                    rA_22 = ct.register_tensor(f16, (8, 4))
                    rA_2221 = ct.register_tensor(f16, (8, 8))
                    fill(rA_2221, f16(0.0))
                    tXrA_21_t = ct.partition_D(rA_21_t, (8, 2))
                    tXsA_21_t = ct.partition_S(sA_21_t, (8, 2))
                    tXsA_22 = ct.partition_S(sA_22, (8, 2))
                    tXrA_22 = ct.partition_D(rA_22, (8, 2))
                    ct.tile_copy((8, 2), tXsA_21_t[:, :, 0], tXrA_21_t[:, :, 0])
                    ct.tile_copy((8, 2), tXsA_22[:, :, 0], tXrA_22[:, :, 0])
                    for k in grid(4, attrs="u+"):
                        mma(tiled_mma_8x8, rA_2221, tXrA_22[:, :, k % 2], tXrA_21_t[:, :, k % 2], rA_2221)
                        if k < 3:
                            ct.tile_copy((8, 2), tXsA_21_t[:, :, k + 1], tXrA_21_t[:, :, (k + 1) % 2])
                            ct.tile_copy((8, 2), tXsA_22[:, :, k + 1], tXrA_22[:, :, (k + 1) % 2])
                    rA_21_inv = f16(-1.0) * rA_2221
                    tXrA_21_inv = ct.partition_S(rA_21_inv, (8, 8))
                    tXsA_21_inv = ct.partition_D(sA_21, (8, 8))
                    ct.tile_copy((8, 8), tXrA_21_inv, tXsA_21_inv)
                syncthreads()

                with for_each_warp((0, 2)) as i:
                    sA_11 = tensor_view(
                        sA[32 * i : 32 * i + 16, 32 * i : 32 * i + 16],
                        TensorLayout((16, 16), (1, 64)),
                        "shared",
                        volatile=True,
                    )
                    sA_21 = tensor_view(
                        sA[32 * i + 16 : 32 * i + 32, 32 * i : 32 * i + 16],
                        TensorLayout((16, 16), (1, 64)),
                        "shared",
                        volatile=True,
                    )
                    sA_22 = tensor_view(
                        sA[32 * i + 16 : 32 * i + 32, 32 * i + 16 : 32 * i + 32],
                        TensorLayout((16, 16), (64, 1)),
                        "shared",
                        volatile=True,
                    )
                    rA_11 = ct.register_tensor(f16, (16, 16))
                    rA_21 = ct.register_tensor(f16, (16, 16))
                    rA_22 = ct.register_tensor(f16, (16, 16))
                    rA_2221 = ct.register_tensor(f32, (16, 16))
                    fill(rA_2221, 0.0)
                    ct.copy(sA_22, rA_22)
                    ct.copy(sA_21, rA_21)
                    mma(tiled_mma_16x16, rA_2221, rA_22, rA_21, rA_2221)
                    rA_2221_f16 = cast(rA_2221, f16)
                    rA_2111 = ct.register_tensor(f32, (16, 16))
                    fill(rA_2111, 0.0)
                    ct.copy(sA_11, rA_11)
                    mma(tiled_mma_16x16, rA_2111, rA_2221_f16, rA_11, rA_2111)
                    rA_21_inv = cast(f32(-1.0) * rA_2111, f16)
                    tXrA_21_inv = ct.partition_S(rA_21_inv, (16, 16))
                    sA_21_t = transpose(sA_21, 1, 0)
                    tXsA_21_inv = ct.partition_D(sA_21_t, (16, 16))
                    ct.tile_copy((16, 16), tXrA_21_inv, tXsA_21_inv)
                syncthreads()

                sA_11 = tensor_view(sA[0:32, 0:32], TensorLayout((32, 32), (1, 64)), "shared", volatile=True)
                sA_21 = tensor_view(sA[32:64, 0:32], TensorLayout((32, 32), (64, 1)), "shared", volatile=True)
                sA_22 = tensor_view(sA[32:64, 32:64], TensorLayout((32, 32), (64, 1)), "shared", volatile=True)
                rA_11 = ct.register_tensor(f16, (32, 32))
                rA_21 = ct.register_tensor(f16, (32, 32))
                rA_2111 = ct.register_tensor(f32, (32, 32))
                fill(rA_2111, 0.0)
                tXrA_11 = ct.partition_D(rA_11, (32, 16))
                tXrA_21 = ct.partition_D(rA_21, (32, 16))
                tXsA_11 = ct.partition_S(sA_11, (32, 16))
                tXsA_21 = ct.partition_S(sA_21, (32, 16))
                ct.tile_copy((32, 16), tXsA_11[:, :, 0], tXrA_11[:, :, 0])
                ct.tile_copy((32, 16), tXsA_21[:, :, 0], tXrA_21[:, :, 0])
                ct.tile_copy((32, 16), tXsA_11[:, :, 1], tXrA_11[:, :, 1])
                ct.tile_copy((32, 16), tXsA_21[:, :, 1], tXrA_21[:, :, 1])
                mma(tiled_mma_32x32, rA_2111, tXrA_21[:, :, 0], tXrA_11[:, :, 0], rA_2111)
                mma(tiled_mma_32x32, rA_2111, tXrA_21[:, :, 1], tXrA_11[:, :, 1], rA_2111)
                rA_2111_f16 = cast(rA_2111, f16)
                tXrA_2111_f16 = ct.partition_S(rA_2111_f16, (32, 32))
                tXsA_2111_f16 = ct.partition_D(sA_21, (32, 32))
                ct.tile_copy((32, 32), tXrA_2111_f16, tXsA_2111_f16)
                sA_21_t = transpose(sA_21, 1, 0)
                rA_21_t = ct.register_tensor(f16, (32, 32))
                rA_22 = ct.register_tensor(f16, (32, 32))
                rA_2221 = ct.register_tensor(f32, (32, 32))
                fill(rA_2221, 0.0)
                tXrA_21_t = ct.partition_D(rA_21_t, (32, 16))
                tXsA_21_t = ct.partition_S(sA_21_t, (32, 16))
                tXrA_22 = ct.partition_D(rA_22, (32, 16))
                tXsA_22 = ct.partition_S(sA_22, (32, 16))
                ct.tile_copy((32, 16), tXsA_21_t[:, :, 0], tXrA_21_t[:, :, 0])
                ct.tile_copy((32, 16), tXsA_22[:, :, 0], tXrA_22[:, :, 0])
                ct.tile_copy((32, 16), tXsA_21_t[:, :, 1], tXrA_21_t[:, :, 1])
                ct.tile_copy((32, 16), tXsA_22[:, :, 1], tXrA_22[:, :, 1])
                mma(tiled_mma_32x32, rA_2221, tXrA_22[:, :, 0], tXrA_21_t[:, :, 0], rA_2221)
                mma(tiled_mma_32x32, rA_2221, tXrA_22[:, :, 1], tXrA_21_t[:, :, 1], rA_2221)
                rA_21_inv = cast(f32(-1.0) * rA_2221, f16)
                tXrA_21_inv = ct.partition_S(rA_21_inv, (32, 32))
                tXsA_21_inv = ct.partition_D(sA_21, (32, 32))
                ct.tile_copy((32, 32), tXrA_21_inv, tXsA_21_inv)

                syncthreads()

        return script_module


def inverse_64x64_kernel():
    from hidet.lang.cute import for_each_thread, for_each_warp
    import hidet.lang.cute as ct
    from hidet.ir.cute import TensorLayout
    from hidet.ir.cute.ops import tensor_view, make_tensor, cast, mma, fill, transpose
    from hidet.ir.dtypes import f16, f32
    from hidet.lang import attrs, tensor_pointer, grid
    from hidet.ir.layout import row_major
    from hidet.lang.cuda import dynamic_shared_memory, syncthreads
    from hidet.ir.primitives.cuda import syncwarp, cp_async_commit_group, cp_async_wait_all
    from hidet.ir.cute import MmaAtom, TiledMma, Level

    rmem_layout = ct.make_register_layout("thread", (1, 1), TensorLayout((1, 1), (1, 1)))
    a = TensorLayout(((1,), (2, 2)), ((1,), (2, 1)))
    b = TensorLayout(((1,), (2, 2)), ((1,), (1, 2)))
    c = TensorLayout(((1,), (2, 2)), ((1,), (2, 1)))
    mma_atom = MmaAtom("thread", (2, 2, 2), a, b, c, c)
    tiled_mma_2x2 = TiledMma(mma_atom, [])
    mma_atom = MmaAtom("thread", (2, 2, 2), a, b, c, c, (2, 2))
    tiled_mma_4x4 = TiledMma(mma_atom, [])

    a = TensorLayout(((1,), (1, 2)), ((1,), (1, 1)))
    b = TensorLayout(((1,), (2, 2)), ((1,), (1, 2)))
    c = TensorLayout(((1,), (1, 2)), ((1,), (1, 1)))
    mma_atom = MmaAtom("thread", (1, 2, 2), a, b, c, c, (1, 1))
    thread_in_warp = Level("thread", "warp", (8, 4), TensorLayout((8, 4), (1, 8)))
    tiled_mma_8x8 = TiledMma(mma_atom, [thread_in_warp])

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    tiled_mma_16x16 = TiledMma(mma_atom, [])

    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (1, 1))
    tiled_mma_32x32 = TiledMma(mma_atom, [warp_in_threadblock])

    with hidet.script_module() as script_module:

        @hidet.script
        def func(A: f16[64, 64], inv_A: f16[64, 64]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128, 1, 1
            attrs.cuda.grid_dim = 1, 1, 1
            attrs.cuda.dynamic_smem_bytes = 64 * 64 * f16.nbytes

            sA = tensor_pointer('float16', shape=[64, 64], layout=row_major(64, 64))
            sA = dynamic_shared_memory(0, f16)

            gA = ct.global_view(A, shape=(64, 64), stride=(64, 1))
            sA_view = tensor_view(sA, TensorLayout((64, 64), (64, 1)), "shared", volatile=True)
            ct.copy(gA, sA_view)
            cp_async_commit_group()
            cp_async_wait_all()
            syncthreads()

            with for_each_thread((0, 64)) as i:
                sA_11 = tensor_view(sA[i : i + 1, i : i + 1], TensorLayout((1, 1), (64, 1)), "shared")
                rA_11 = make_tensor(f16, rmem_layout, "register")
                ct.copy(sA_11, rA_11)
                rA_11 = cast(f32(1.0) / cast(rA_11, f32), f16)
                ct.copy(rA_11, sA_11)
            syncthreads()

            with for_each_thread((0, 32)) as i:
                sA_11 = tensor_view(sA[2 * i : 2 * i + 1, 2 * i : 2 * i + 1], TensorLayout((1, 1), (64, 1)), "shared")
                rA_11 = make_tensor(f16, rmem_layout, "register")
                ct.copy(sA_11, rA_11)
                sA_21 = tensor_view(
                    sA[2 * i + 1 : 2 * i + 2, 2 * i : 2 * i + 1], TensorLayout((1, 1), (64, 1)), "shared"
                )
                rA_21 = ct.register_tensor(f16, (1, 1))
                ct.copy(sA_21, rA_21)
                sA_22 = tensor_view(
                    sA[2 * i + 1 : 2 * i + 2, 2 * i + 1 : 2 * i + 2], TensorLayout((1, 1), (64, 1)), "shared"
                )
                rA_22 = ct.register_tensor(f16, (1, 1))
                ct.copy(sA_22, rA_22)
                rA_21_inv = cast(f32(-1.0) * cast(rA_22, f32) * cast(rA_21, f32) * cast(rA_11, f32), f16)
                tXrA_21_inv = ct.partition_S(rA_21_inv, (1, 1))
                tXsA_21 = ct.partition_D(sA_21, (1, 1))
                ct.tile_copy((1, 1), tXrA_21_inv, tXsA_21)
            syncwarp()

            with for_each_thread((0, 16)) as i:
                sA_11 = tensor_view(sA[4 * i : 4 * i + 2, 4 * i : 4 * i + 2], TensorLayout((2, 2), (1, 64)), "shared")
                sA_21 = tensor_view(
                    sA[4 * i + 2 : 4 * i + 4, 4 * i : 4 * i + 2], TensorLayout((2, 2), (64, 1)), "shared"
                )
                sA_22 = tensor_view(
                    sA[4 * i + 2 : 4 * i + 4, 4 * i + 2 : 4 * i + 4], TensorLayout((2, 2), (64, 1)), "shared"
                )
                rA_11 = ct.register_tensor(f16, (2, 2))
                rA_21 = ct.register_tensor(f16, (2, 2))
                rA_2111 = ct.register_tensor(f16, (2, 2))
                fill(rA_2111, f16(0.0))
                ct.copy(sA_21, rA_21)
                ct.copy(sA_11, rA_11)
                mma(tiled_mma_2x2, rA_2111, rA_21, rA_11, rA_2111)
                ct.copy(rA_2111, sA_21)
                sA_21_t = transpose(sA_21, 1, 0)
                rA_21_t = ct.register_tensor(f16, (2, 2))
                rA_22 = ct.register_tensor(f16, (2, 2))
                rA_2221 = ct.register_tensor(f16, (2, 2))
                fill(rA_2221, f16(0.0))
                ct.copy(sA_22, rA_22)
                ct.copy(sA_21_t, rA_21_t)
                mma(tiled_mma_2x2, rA_2221, rA_22, rA_21_t, rA_2221)
                rA_21_inv = f16(-1.0) * rA_2221
                tXrA_21_inv = ct.partition_S(rA_21_inv, (2, 2))
                tXsA_21 = ct.partition_D(sA_21, (2, 2))
                ct.tile_copy((2, 2), tXrA_21_inv, tXsA_21)
            syncwarp()

            with for_each_thread((0, 8)) as i:
                sA_11 = tensor_view(sA[8 * i : 8 * i + 4, 8 * i : 8 * i + 4], TensorLayout((4, 4), (1, 64)), "shared")
                sA_21 = tensor_view(
                    sA[8 * i + 4 : 8 * i + 8, 8 * i : 8 * i + 4], TensorLayout((4, 4), (64, 1)), "shared"
                )
                sA_22 = tensor_view(
                    sA[8 * i + 4 : 8 * i + 8, 8 * i + 4 : 8 * i + 8], TensorLayout((4, 4), (64, 1)), "shared"
                )
                rA_11 = ct.register_tensor(f16, (4, 4))
                rA_21 = ct.register_tensor(f16, (4, 4))
                rA_2111 = ct.register_tensor(f16, (4, 4))
                fill(rA_2111, f16(0.0))
                tXrA_11 = ct.partition_D(rA_11, (4, 2))
                tXrA_21 = ct.partition_D(rA_21, (4, 2))
                tXsA_11 = ct.partition_S(sA_11, (4, 2))
                tXsA_21 = ct.partition_S(sA_21, (4, 2))
                ct.tile_copy((4, 2), tXsA_11[:, :, 0], tXrA_11[:, :, 0])
                ct.tile_copy((4, 2), tXsA_21[:, :, 0], tXrA_21[:, :, 0])
                ct.tile_copy((4, 2), tXsA_11[:, :, 1], tXrA_11[:, :, 1])
                ct.tile_copy((4, 2), tXsA_21[:, :, 1], tXrA_21[:, :, 1])
                mma(tiled_mma_4x4, rA_2111, tXrA_21[:, :, 0], tXrA_11[:, :, 0], rA_2111)
                mma(tiled_mma_4x4, rA_2111, tXrA_21[:, :, 1], tXrA_11[:, :, 1], rA_2111)
                sA_21_t = transpose(sA_21, 1, 0)
                rA_21_t = ct.register_tensor(f16, (4, 4))
                rA_22 = ct.register_tensor(f16, (4, 4))
                rA_2221 = ct.register_tensor(f16, (4, 4))
                fill(rA_2221, f16(0.0))
                tXrA_21_t = ct.partition_D(rA_21_t, (4, 2))
                tXsA_21_t = ct.partition_S(sA_21_t, (4, 2))
                tXrA_22 = ct.partition_D(rA_22, (4, 2))
                tXsA_22 = ct.partition_S(sA_22, (4, 2))
                ct.tile_copy((4, 2), tXsA_21_t[:, :, 0], tXrA_21_t[:, :, 0])
                ct.tile_copy((4, 2), tXsA_22[:, :, 0], tXrA_22[:, :, 0])
                ct.tile_copy((4, 2), tXsA_21_t[:, :, 1], tXrA_21_t[:, :, 1])
                ct.tile_copy((4, 2), tXsA_22[:, :, 1], tXrA_22[:, :, 1])
                mma(tiled_mma_4x4, rA_2221, tXrA_22[:, :, 0], tXrA_21_t[:, :, 0], rA_2221)
                mma(tiled_mma_4x4, rA_2221, tXrA_22[:, :, 1], tXrA_21_t[:, :, 1], rA_2221)
                rA_21_inv = f16(-1.0) * rA_2221
                tXrA_21_inv = ct.partition_S(rA_21_inv, (4, 4))
                tXsA_21_inv = ct.partition_D(sA_21, (4, 4))
                ct.tile_copy((4, 4), tXrA_21_inv, tXsA_21_inv)
            syncthreads()

            with for_each_warp((0, 4)) as i:
                sA_11 = tensor_view(
                    sA[16 * i : 16 * i + 8, 16 * i : 16 * i + 8], TensorLayout((8, 8), (1, 64)), "shared", volatile=True
                )
                sA_21 = tensor_view(
                    sA[16 * i + 8 : 16 * i + 16, 16 * i : 16 * i + 8],
                    TensorLayout((8, 8), (64, 1)),
                    "shared",
                    volatile=True,
                )
                rA_11 = ct.register_tensor(f16, (8, 4))
                rA_21 = ct.register_tensor(f16, (8, 4))
                rA_2111 = ct.register_tensor(f16, (8, 8))
                fill(rA_2111, f16(0.0))
                tXrA_11 = ct.partition_D(rA_11, (8, 2))
                tXrA_21 = ct.partition_D(rA_21, (8, 2))
                tXsA_11 = ct.partition_S(sA_11, (8, 2))
                tXsA_21 = ct.partition_S(sA_21, (8, 2))
                ct.tile_copy((8, 2), tXsA_11[:, :, 0], tXrA_11[:, :, 0])
                ct.tile_copy((8, 2), tXsA_21[:, :, 0], tXrA_21[:, :, 0])
                for k in grid(4, attrs="u+"):
                    mma(tiled_mma_8x8, rA_2111, tXrA_21[:, :, k % 2], tXrA_11[:, :, k % 2], rA_2111)
                    if k < 3:
                        ct.tile_copy((8, 2), tXsA_11[:, :, k + 1], tXrA_11[:, :, (k + 1) % 2])
                        ct.tile_copy((8, 2), tXsA_21[:, :, k + 1], tXrA_21[:, :, (k + 1) % 2])
                ct.copy(rA_2111, sA_21)
                sA_21_t = transpose(sA_21, 1, 0)
                sA_22 = tensor_view(
                    sA[16 * i + 8 : 16 * i + 16, 16 * i + 8 : 16 * i + 16],
                    TensorLayout((8, 8), (64, 1)),
                    "shared",
                    volatile=True,
                )
                rA_21_t = ct.register_tensor(f16, (8, 4))
                rA_22 = ct.register_tensor(f16, (8, 4))
                rA_2221 = ct.register_tensor(f16, (8, 8))
                fill(rA_2221, f16(0.0))
                tXrA_21_t = ct.partition_D(rA_21_t, (8, 2))
                tXsA_21_t = ct.partition_S(sA_21_t, (8, 2))
                tXsA_22 = ct.partition_S(sA_22, (8, 2))
                tXrA_22 = ct.partition_D(rA_22, (8, 2))
                ct.tile_copy((8, 2), tXsA_21_t[:, :, 0], tXrA_21_t[:, :, 0])
                ct.tile_copy((8, 2), tXsA_22[:, :, 0], tXrA_22[:, :, 0])
                for k in grid(4, attrs="u+"):
                    mma(tiled_mma_8x8, rA_2221, tXrA_22[:, :, k % 2], tXrA_21_t[:, :, k % 2], rA_2221)
                    if k < 3:
                        ct.tile_copy((8, 2), tXsA_21_t[:, :, k + 1], tXrA_21_t[:, :, (k + 1) % 2])
                        ct.tile_copy((8, 2), tXsA_22[:, :, k + 1], tXrA_22[:, :, (k + 1) % 2])
                rA_21_inv = f16(-1.0) * rA_2221
                tXrA_21_inv = ct.partition_S(rA_21_inv, (8, 8))
                tXsA_21_inv = ct.partition_D(sA_21, (8, 8))
                ct.tile_copy((8, 8), tXrA_21_inv, tXsA_21_inv)
            syncthreads()

            with for_each_warp((0, 2)) as i:
                sA_11 = tensor_view(
                    sA[32 * i : 32 * i + 16, 32 * i : 32 * i + 16],
                    TensorLayout((16, 16), (1, 64)),
                    "shared",
                    volatile=True,
                )
                sA_21 = tensor_view(
                    sA[32 * i + 16 : 32 * i + 32, 32 * i : 32 * i + 16],
                    TensorLayout((16, 16), (1, 64)),
                    "shared",
                    volatile=True,
                )
                sA_22 = tensor_view(
                    sA[32 * i + 16 : 32 * i + 32, 32 * i + 16 : 32 * i + 32],
                    TensorLayout((16, 16), (64, 1)),
                    "shared",
                    volatile=True,
                )
                rA_11 = ct.register_tensor(f16, (16, 16))
                rA_21 = ct.register_tensor(f16, (16, 16))
                rA_22 = ct.register_tensor(f16, (16, 16))
                rA_2221 = ct.register_tensor(f32, (16, 16))
                fill(rA_2221, 0.0)
                ct.copy(sA_22, rA_22)
                ct.copy(sA_21, rA_21)
                mma(tiled_mma_16x16, rA_2221, rA_22, rA_21, rA_2221)
                rA_2221_f16 = cast(rA_2221, f16)
                rA_2111 = ct.register_tensor(f32, (16, 16))
                fill(rA_2111, 0.0)
                ct.copy(sA_11, rA_11)
                mma(tiled_mma_16x16, rA_2111, rA_2221_f16, rA_11, rA_2111)
                rA_21_inv = cast(f32(-1.0) * rA_2111, f16)
                tXrA_21_inv = ct.partition_S(rA_21_inv, (16, 16))
                sA_21_t = transpose(sA_21, 1, 0)
                tXsA_21_inv = ct.partition_D(sA_21_t, (16, 16))
                ct.tile_copy((16, 16), tXrA_21_inv, tXsA_21_inv)
            syncthreads()

            sA_11 = tensor_view(sA[0:32, 0:32], TensorLayout((32, 32), (1, 64)), "shared", volatile=True)
            sA_21 = tensor_view(sA[32:64, 0:32], TensorLayout((32, 32), (64, 1)), "shared", volatile=True)
            sA_22 = tensor_view(sA[32:64, 32:64], TensorLayout((32, 32), (64, 1)), "shared", volatile=True)
            rA_11 = ct.register_tensor(f16, (32, 32))
            rA_21 = ct.register_tensor(f16, (32, 32))
            rA_2111 = ct.register_tensor(f32, (32, 32))
            fill(rA_2111, 0.0)
            tXrA_11 = ct.partition_D(rA_11, (32, 16))
            tXrA_21 = ct.partition_D(rA_21, (32, 16))
            tXsA_11 = ct.partition_S(sA_11, (32, 16))
            tXsA_21 = ct.partition_S(sA_21, (32, 16))
            ct.tile_copy((32, 16), tXsA_11[:, :, 0], tXrA_11[:, :, 0])
            ct.tile_copy((32, 16), tXsA_21[:, :, 0], tXrA_21[:, :, 0])
            ct.tile_copy((32, 16), tXsA_11[:, :, 1], tXrA_11[:, :, 1])
            ct.tile_copy((32, 16), tXsA_21[:, :, 1], tXrA_21[:, :, 1])
            mma(tiled_mma_32x32, rA_2111, tXrA_21[:, :, 0], tXrA_11[:, :, 0], rA_2111)
            mma(tiled_mma_32x32, rA_2111, tXrA_21[:, :, 1], tXrA_11[:, :, 1], rA_2111)
            rA_2111_f16 = cast(rA_2111, f16)
            tXrA_2111_f16 = ct.partition_S(rA_2111_f16, (32, 32))
            tXsA_2111_f16 = ct.partition_D(sA_21, (32, 32))
            ct.tile_copy((32, 32), tXrA_2111_f16, tXsA_2111_f16)
            sA_21_t = transpose(sA_21, 1, 0)
            rA_21_t = ct.register_tensor(f16, (32, 32))
            rA_22 = ct.register_tensor(f16, (32, 32))
            rA_2221 = ct.register_tensor(f32, (32, 32))
            fill(rA_2221, 0.0)
            tXrA_21_t = ct.partition_D(rA_21_t, (32, 16))
            tXsA_21_t = ct.partition_S(sA_21_t, (32, 16))
            tXrA_22 = ct.partition_D(rA_22, (32, 16))
            tXsA_22 = ct.partition_S(sA_22, (32, 16))
            ct.tile_copy((32, 16), tXsA_21_t[:, :, 0], tXrA_21_t[:, :, 0])
            ct.tile_copy((32, 16), tXsA_22[:, :, 0], tXrA_22[:, :, 0])
            ct.tile_copy((32, 16), tXsA_21_t[:, :, 1], tXrA_21_t[:, :, 1])
            ct.tile_copy((32, 16), tXsA_22[:, :, 1], tXrA_22[:, :, 1])
            mma(tiled_mma_32x32, rA_2221, tXrA_22[:, :, 0], tXrA_21_t[:, :, 0], rA_2221)
            mma(tiled_mma_32x32, rA_2221, tXrA_22[:, :, 1], tXrA_21_t[:, :, 1], rA_2221)
            rA_21_inv = cast(f32(-1.0) * rA_2221, f16)
            tXrA_21_inv = ct.partition_S(rA_21_inv, (32, 32))
            tXsA_21_inv = ct.partition_D(sA_21, (32, 32))
            ct.tile_copy((32, 32), tXrA_21_inv, tXsA_21_inv)

            syncthreads()
            gA_inv = ct.global_view(inv_A, shape=(64, 64), stride=(64, 1))
            ct.copy(sA_view, gA_inv)

    func = script_module.build()
    return func


def check_inverse_64x64_tile():
    with hidet.option.context():
        hidet.option.cache_dir('./gdn_attention')
        hidet.option.search_space(2)
        hidet.option.debug_cache_tuning()
        hidet.option.save_lower_ir(True)
        func = inverse_64x64_kernel()

        diagonal = torch.eye(64, dtype=torch.float16)
        lower_triangular = torch.tril(
            torch.randint(low=-3, high=3, size=(64, 64), dtype=torch.float16) / 64.0, diagonal=-1
        )
        A = diagonal + lower_triangular
        A = A.cuda()
        print(A)
        A_inv = torch.linalg.inv(A.to(torch.float32))
        print(A.to(torch.float32) @ A_inv)
        A_inv_hidet = torch.empty_like(A)
        func(A, A_inv_hidet)
        torch.cuda.synchronize()
        print(A.to(torch.float32) @ A_inv_hidet.to(torch.float32))
        import numpy as np

        np.testing.assert_allclose(A_inv_hidet.to(torch.float32).cpu().numpy(), A_inv.cpu().numpy(), atol=1, rtol=1e-2)


def main():
    with hidet.option.context():
        hidet.option.cache_dir('./gdn_attention')
        hidet.option.search_space(2)
        hidet.option.debug_cache_tuning()
        hidet.option.save_lower_ir(True)

        gdn = GDNAttention(H=64, Hg=64, K=64)

        func = gdn._experimental()


if __name__ == '__main__':
    main()
