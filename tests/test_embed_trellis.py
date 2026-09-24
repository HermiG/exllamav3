import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fork package must win over any installed copy
import platform
import shutil
import tempfile
import types
from unittest import mock
try:
    import pytest  # noqa: F401  (file is pytest-compatible; also runnable as a script)
except ImportError:
    pytest = None
import torch

from exllamav3.modules.quant.exl3_lib import embed_trellis as et
from exllamav3.modules.quant.exl3_lib.embed_trellis import (
    pack_rows, unpack_rows, dequant_rows, lcg_signs, hadamard_butterfly,
    dequant_rows_transformed, forward_transform, quantize_rows_grouped, mul1_codebook,
    GROUP,
)
from exllamav3.conversion.ngram import StreamingSafetensorsWriter

"""
Self-tests for the trellis-quantized token embedding (exl3_trellis_embed): the reference
codec (exllamav3/modules/quant/exl3_lib/embed_trellis.py), the fused device-mapped gather
kernel (ext.trellis_embed_*), the Embedding module branch, and the conversion engine round-
trip. All bit-exactness claims are between the paths themselves (codec vs. fused kernel vs.
module); no external corpus or model is required. Importing the file requires a working
exllamav3 build (precompiled ext); GPU-gated tests skip when CUDA is unavailable.
"""

cuda = torch.cuda.is_available()
dev = "cuda:0"


def _skip_no_cuda():
    """Real SKIP under pytest (a bare `return` records PASS - the GPU contract tests must
    not silently green on CPU-only CI); as a script, pass through to the caller's return."""
    if pytest is not None and __name__ != "__main__":
        pytest.skip("CUDA unavailable")



def _skip_no_avx2():
    """Tests that need SOME encoded table via the conversion engine's CPU AVX2 Viterbi
    encoder (exllamav3.conversion.embed.encode_rows, JIT-compiled with -march=x86-64-v3):
    real SKIP on non-x86-64 hosts (incl. CUDA-equipped ARM), where the encoder raises
    instead of skipping. The platform check runs FIRST: skipping unconditionally under
    pytest would drop these tests on every host, x86-64 included."""
    if platform.machine() not in {"x86_64", "AMD64"}:
        if pytest is not None and __name__ != "__main__":
            pytest.skip("AVX2 x86-64 encoder host required")
        return True
    return False


def _synth_table(N: int, D: int, K: int, seed: int, scale_rows: bool = True):
    """Synthetic packed table + column scales: random rows, quantized through the codec."""
    g = torch.Generator().manual_seed(11)
    w = torch.randn(N, D, generator = g)
    if scale_rows:
        w = w * torch.randn(D, generator = g).abs().clamp(0.05).unsqueeze(0)
    w = w.half().float()
    ids = torch.arange(N, dtype = torch.int64)
    cs = (torch.rand(D, generator = g) * 0.6 + 0.6).half()
    y = forward_transform(w, cs.float(), seed, ids)
    packed, _ = quantize_rows_grouped(y, K, D, _cpu_encode, group = GROUP)
    return w, packed, cs


def _cpu_encode(w_pre: torch.Tensor, K: int, D: int) -> torch.Tensor:
    """CPU trellis encoder for the tests: the fused dequant is the codec's decode chain, so
    the tests need only SOME valid state stream; delegate to the conversion engine's AVX2
    encoder (exllamav3.conversion.embed.encode_rows). If the import or the call fails, the
    exception is swallowed and an AssertionError is raised - there is no pure-torch
    fallback."""
    try:
        from exllamav3.conversion.embed import encode_rows
        return encode_rows(w_pre, K, D, threads = 4)
    except Exception as e:
        raise AssertionError("conversion.embed.encode_rows must be importable for tests") from e


@torch.inference_mode()
def test_codec_pack_unpack_roundtrip():
    """pack_rows consumes 16-bit ring states (low K bits = the position's new symbol) and
    unpack_rows reconstructs exactly those ring windows - the round-trip is only exact for
    consistent ring states, which is what the encoder emits; build them from fresh K-bit
    symbols with the same roll-accumulate the decoder side uses. The ring semantics are
    pinned by an independent oracle that slices the packed bitstream directly."""
    for K in (4, 5, 6, 7, 8):
        for D in (512, 5120):
            G = D // GROUP
            g = torch.Generator().manual_seed(K * 7 + D)
            new = torch.randint(0, 1 << K, (37, D), generator = g, dtype = torch.int16)
            states = new.to(torch.int64)
            shift, lag = K, 1
            while shift < 16:
                width = min(K, 16 - shift)
                states = states | ((torch.roll(new, lag, dims = 1).to(torch.int64)) & ((1 << width) - 1)) << shift
                shift += K
                lag += 1
            scales = (torch.rand(37, G, generator = g) + 0.5).half()
            packed = pack_rows(states, scales, K, D, G)
            assert packed.dtype == torch.int16
            assert packed.shape == (37, G + D * K // 16)
            # independent ring oracle: extract each position's K-bit code k_i straight
            # from the packed words (stream bits [i*K, (i+1)*K)), then combine with the
            # ring recurrence's closed form - state_i = k_i | k_{i-1} << K |
            # (k_{i-2} & m2) << 2*K | (k_{i-3} & m3) << 3*K (the form the fused kernel uses).
            # Bit shifts/masks on the packed representation only - no roll, no unpack_rows.
            words = packed[:, G:].view(torch.uint16).to(torch.int64)
            W = words.shape[1]
            t0 = torch.arange(D, dtype = torch.int64) * K
            o = t0 & 15
            w0 = t0 // 16
            o2 = (o + K - 16).clamp(min = 0)
            k = (words[:, w0] >> o) & ((1 << K) - 1)
            k = k | ((words[:, (w0 + 1) % W] & ((1 << o2) - 1)) << (16 - o))
            idx = torch.arange(D, dtype = torch.int64)
            m2 = (1 << (16 - 2 * K)) - 1
            m3 = (1 << (16 - 3 * K)) - 1 if 16 - 3 * K > 0 else 0
            ref = k | (k[:, (idx - 1) % D] << K) | ((k[:, (idx - 2) % D] & m2) << (2 * K)) \
                | ((k[:, (idx - 3) % D] & m3) << (3 * K))
            assert torch.equal(ref, states), f"packed stream != ring states at K = {K}, D = {D}"
            back, sc = unpack_rows(packed, K, D, G)
            assert torch.equal(back, states), f"pack/unpack round-trip broken at K = {K}, D = {D}"
            assert torch.equal(sc, scales), "scale words not preserved"

    # absolute byte pin (little-endian uint16 words; stream bit (i*K + m) = bit m of
    # state_i, word 0 = the fp16 scale word's bit pattern): one row re-implemented in
    # pure Python ints - independent of every codec tensor op, so a byte-order or
    # bit-order regression in pack_rows fails here
    K, D, G = 4, 256, 1
    g = torch.Generator().manual_seed(999)
    states = torch.randint(-32768, 32768, (1, D), generator = g, dtype = torch.int16).to(torch.int64)
    packed = pack_rows(states, torch.tensor([1.5], dtype = torch.float16), K, D, G)
    exp_words = [torch.tensor([1.5], dtype = torch.float16).view(torch.int16).item()]
    exp_words += [0] * (D * K // 16)
    for i in range(D):
        for m in range(K):
            bit = (states[0, i].item() >> m) & 1
            b = i * K + m
            exp_words[b // 16 + 1] |= bit << (b % 16)
    assert packed[0].view(torch.uint16).tolist() == exp_words, \
        "packed bytes != pure-Python little-endian reference"


@torch.inference_mode()
def test_codec_multiblock_partition_invariance():
    """pack_rows/unpack_rows/dequant_rows process rows in PLANE_BUDGET blocks and claim
    the blocking is value- and bit-identical: decoding N rows at once must therefore be
    bit-identical to decoding the rows in two halves (the second half starts mid-block,
    so any off-by-one in the block slicing shows up). N crosses several blocks in every
    pass (tightest: dequant_rows' PLANE_BUDGET // D = 819 rows/block at D = 5120). No
    AVX2 encoder needed: the states are built from fresh K-bit symbols with the same
    roll-accumulate as the round-trip test (valid ring states, arbitrary values)."""
    N, D = 900, 5120
    G = D // GROUP
    for K in (4, 8):
        g = torch.Generator().manual_seed(K * 13 + D)
        new = torch.randint(0, 1 << K, (N, D), generator = g, dtype = torch.int16)
        states = new.to(torch.int64)
        shift, lag = K, 1
        while shift < 16:
            width = min(K, 16 - shift)
            states = states | ((torch.roll(new, lag, dims = 1).to(torch.int64)) & ((1 << width) - 1)) << shift
            shift += K
            lag += 1
        scales = (torch.rand(N, G, generator = g) + 0.5).half()
        cs = (torch.rand(D, generator = g) * 0.6 + 0.6).half()
        packed = pack_rows(states, scales, K, D, G)
        ids = torch.arange(N, dtype = torch.int64)
        cb = mul1_codebook("cpu")
        h = N // 2
        whole = [
            unpack_rows(packed, K, D, G)[0],
            dequant_rows(packed, K, cb, D, G),
            dequant_rows_transformed(packed, cs, K, 7, D, G, row_ids = ids),
        ]
        halves = [
            torch.cat([unpack_rows(packed[:h], K, D, G)[0], unpack_rows(packed[h:], K, D, G)[0]]),
            torch.cat([dequant_rows(packed[:h], K, cb, D, G), dequant_rows(packed[h:], K, cb, D, G)]),
            torch.cat([dequant_rows_transformed(packed[:h], cs, K, 7, D, G, row_ids = ids[:h]),
                       dequant_rows_transformed(packed[h:], cs, K, 7, D, G, row_ids = ids[h:])]),
        ]
        for a, b in zip(whole, halves):
            assert torch.equal(a, b), f"row-blocking changed decode results at K = {K}"


@torch.inference_mode()
def test_codec_pad_rows_decode_to_zero():
    """All-zero state words (pad rows) must decode to exactly 0.0: zero codeword + zero
    ring contribution, no sign or scale leakage."""
    for K in (4, 5, 6, 7, 8):
        D, G = 512, 2
        packed = torch.zeros(4, et.words_per_row(D, K, G), dtype = torch.int16)
        cs = torch.ones(D, dtype = torch.float16)
        x = dequant_rows_transformed(packed, cs, K, 0, D, G, row_ids = torch.arange(4))
        assert torch.equal(x, torch.zeros(4, D)), f"pad rows not exactly zero at K = {K}"



@torch.inference_mode()
def test_encoder_zero_rows_pack_to_zero():
    """All-zero source rows (vocab pad slots) must PACK to all-zero words through the
    encode chain (not just decode to zero when pre-zeroed): the group_prescale
    empty-group guard would otherwise store a 1.0 scale and emit codebook garbage."""
    for K in (4, 5, 6, 7, 8):
        D, G = 512, 2
        w = torch.zeros(3, D)
        w[1] = 1.0  # nonzero neighbor stays on the normal encode path
        packed, scale0 = et.quantize_rows_grouped(
            w, K, D, lambda wp, k, d: torch.zeros(wp.shape[0], d, dtype = torch.int16))
        assert torch.equal(packed[0], torch.zeros_like(packed[0])), f"zero row not zero-packed at K = {K}"
        assert torch.equal(packed[2], torch.zeros_like(packed[2])), f"zero row not zero-packed at K = {K}"
        assert bool((scale0[[0, 2]] == 0).all()), f"zero row scale words not zero at K = {K}"
        assert bool((scale0[1] != 0).all()), f"nonzero row scales clobbered at K = {K}"
        x = dequant_rows_transformed(packed[[0, 2]], torch.ones(D, dtype = torch.float16),
                                     K, 7, D, G, row_ids = torch.tensor([5, 9999]))
        assert torch.equal(x, torch.zeros(2, D)), f"zero-packed rows decode nonzero at K = {K}"

@torch.inference_mode()
def test_lcg_signs_prefix_consistency():
    """Sign stream for row r must be the same stream regardless of the batch it is drawn
    in (the runtime draws D signs per row independently; kernel draws them inlined)."""
    ids = torch.tensor([0, 1, 17, 248319, 2**31], dtype = torch.int64)
    s_all = lcg_signs(ids, 5120, 12345)
    for i in range(ids.shape[0]):
        s_one = lcg_signs(ids[i:i + 1], 5120, 12345)
        assert torch.equal(s_all[i:i + 1], s_one), f"LCG stream row {int(ids[i])} batch-dependent"


@torch.inference_mode()
def test_lcg_signs_match_scalar_stream():
    """The vectorized affine fast-forward must equal the direct per-element uint64 LCG
    stream (the format-pinned sign stream) - runs on CPU, but the file's imports require a
    working exllamav3 build (precompiled ext); no GPU needed."""
    m64 = (1 << 64) - 1
    ids = torch.tensor([0, 1, 511, 2**32 + 7, 2**62], dtype = torch.int64)
    for seed in (0, 12345):
        s = lcg_signs(ids, 512, seed)
        for n in range(ids.shape[0]):
            r = int(ids[n])
            lcg = (seed * et.LCG_KEY_SEED_MUL + r * et.LCG_KEY_ROW_MUL + 1) & m64
            ref = []
            for _ in range(512):
                lcg = (lcg * et.LCG_STEP_MUL + et.LCG_STEP_ADD) & m64
                ref.append(-1.0 if lcg >> 63 else 1.0)
            assert torch.equal(s[n], torch.tensor(ref)), \
                f"LCG fast-forward != scalar stream (row {r}, seed {seed})"


@torch.inference_mode()
def test_hadamard_self_inverse_scaled():
    """H/16 applied twice is the identity in exact arithmetic; in fp32 the decode chain's
    butterfly stage order is what the kernel mirrors, so check the scale-normalized
    orthonormality property instead: columns of H/16 are orthonormal (H H^T = 256 I)."""
    H = torch.zeros(GROUP, GROUP)
    for i in range(GROUP):
        e = torch.zeros(1, GROUP)
        e[0, i] = 1.0
        H[i] = hadamard_butterfly(e, GROUP)[0]
    Hn = H / 16.0
    eye = Hn @ Hn.T
    assert torch.allclose(eye, torch.eye(GROUP), atol = 1e-5, rtol = 0), "H/16 not orthonormal"
    # and the butterfly is its own inverse up to the 1/16 scale, elementwise in exact int
    v = torch.randint(-64, 64, (1, GROUP), dtype = torch.float32)
    hh = hadamard_butterfly(hadamard_butterfly(v, GROUP), GROUP)
    assert torch.equal(hh, v * 256.0), "butterfly twice != 256x (stage order not self-inverse)"


@torch.inference_mode()
def test_fused_kernel_bit_exact_vs_codec():
    """The fused device-mapped gather kernel must reproduce the torch reference decode
    chain bit-exactly for every K, including ring wrap (last -> first position) and the
    byte-window boundary lanes."""
    if not cuda:
        _skip_no_cuda()
        return
    if _skip_no_avx2():
        return
    from exllamav3.ext import exllamav3_ext as ext
    torch.manual_seed(5)
    N = 131
    for K in (4, 5, 6, 7, 8):
        for D in (5120, 512):        # D = 512 -> G = 2 -> 2G % 8 != 0 -> A8 = false for every K
            G = D // GROUP
            for seed in (0, 991):
                w, packed, cs = _synth_table(N, D, K, seed)
                ids = torch.cat([torch.arange(8), torch.arange(N - 8, N),
                                 torch.tensor([G, 127, 128, 129, 65535 % N if N > 65535 else N // 2])])
                ids = ids[ids < N]
                ref = dequant_rows_transformed(packed[ids], cs, K, seed, D, G, row_ids = ids)
                table = packed.pin_memory()
                ptr = ext.trellis_embed_register(table)
                try:
                    out = torch.empty(ids.shape[0], D, dtype = torch.float32, device = dev)
                    # A8 (the aligned 8-byte K = 8 load) is gated on the table base AND (2G)/(2W) 8-alignment
                    # (trellis_embed.cu:414), so D = 512 routes every K - incl. load_codes_bits<8> - through the
                    # A8 = false window even though pin_memory() bases are 8-aligned; misaligned views land here too.
                    ext.trellis_embed_gather(ptr, mul1_codebook(dev), cs.float().to(dev),
                                             ids.to(dev), K, seed, N, out)
                    assert torch.equal(out.cpu(), ref), f"fused kernel != codec at K = {K}, D = {D}, seed = {seed}"
                finally:
                    ext.trellis_embed_unregister(table)


@torch.inference_mode()
def test_register_unregister_roundtrip():
    """Register must be idempotent and return a stable device pointer (refcounted: every
    register() needs its own unregister()); unregister must leave the tensor usable and
    re-registration must succeed (converter + module reload flow)."""
    if not cuda:
        _skip_no_cuda()
        return
    from exllamav3.ext import exllamav3_ext as ext
    t = torch.randint(-32768, 32767, (64, 1024), dtype = torch.int16).pin_memory()
    p1 = ext.trellis_embed_register(t)
    p2 = ext.trellis_embed_register(t)
    assert p1 == p2 and p1 != 0
    ext.trellis_embed_unregister(t)   # refcount: pairs the duplicate register()
    ext.trellis_embed_unregister(t)   # last ref gone: registration torn down here
    p3 = ext.trellis_embed_register(t)
    assert p3 != 0
    ext.trellis_embed_unregister(t)
    _ = t.float() + 1.0   # tensor still fully usable after unregister


@torch.inference_mode()
def test_kernel_oob_id_clamped():
    """Out-of-range ids must not fault the process: an OOB id would be an out-of-bounds
    read of the device-mapped host table (not a trapped fault), so the kernel clamps it
    to the last row - an OOB id decodes bit-identically to row N-1. Loud rejection lives
    in the Python caller (Embedding._gather raises IndexError for host-resident ids
    before the H2D copy). The launcher's host-side validation (unregistered table_ptr,
    oversized n_rows) must raise cleanly, with no sync and no launch."""
    if not cuda:
        _skip_no_cuda()
        return
    if _skip_no_avx2():
        return
    from exllamav3.ext import exllamav3_ext as ext
    N, D, K = 64, 512, 8
    _, packed, cs = _synth_table(N, D, K, 1)
    table = packed.pin_memory()
    ptr = ext.trellis_embed_register(table)
    try:
        cb = mul1_codebook(dev)
        csf = cs.float().to(dev)
        ref = torch.empty(1, D, dtype = torch.float32, device = dev)
        ext.trellis_embed_gather(ptr, cb, csf, torch.tensor([N - 1], dtype = torch.int64, device = dev),
                                 K, 0, N, ref)
        for bad in (-1, N):
            out = torch.empty(1, D, dtype = torch.float32, device = dev)
            ext.trellis_embed_gather(ptr, cb, csf, torch.tensor([bad], dtype = torch.int64, device = dev),
                                     K, 0, N, out)
            assert torch.equal(out, ref), f"OOB id {bad} must clamp to the last row"
        if pytest is not None:   # script mode without pytest: pytest.raises unavailable
            out = torch.empty(1, D, dtype = torch.float32, device = dev)
            with pytest.raises(RuntimeError, match = "not returned by trellis_embed_register"):
                ext.trellis_embed_gather(0xdeadbeef0, cb, csf,
                                         torch.tensor([0], dtype = torch.int64, device = dev),
                                         K, 0, N, out)
            with pytest.raises(RuntimeError, match = "exceeds the registered table size"):
                ext.trellis_embed_gather(ptr, cb, csf,
                                         torch.tensor([0], dtype = torch.int64, device = dev),
                                         K, 0, N + 1, out)
    finally:
        ext.trellis_embed_unregister(table)


@torch.inference_mode()
def test_kernel_edge_cases():
    """Kernel edges not covered by the random-table bit-exact test: a single-row gather
    (n = 1), a G = 1 table (D = 256 - never through the kernel in the other tests), and
    the max 16-bit ring state 0xFFFF (K = 8, all code bytes 0xFF - the last mul1 LUT
    entry). The table is hand-packed (no AVX2 encoder): one fp16 scale word + the
    all-ones code bitstream; the contract is kernel == the torch reference decode of the
    same bytes. D = 256 also keeps 2W % 8 != 0, so K = 8 routes through the
    load_codes_bits window, not the A8 8-byte load."""
    if not cuda:
        _skip_no_cuda()
        return
    from exllamav3.ext import exllamav3_ext as ext
    K, D, G = 8, 256, 1
    W = G + D * K // 16
    for scale, seed in ((1.0, 0), (0.5, 991)):
        sw = torch.tensor([scale], dtype = torch.float16).view(torch.int16).item()
        row = torch.full((1, W), -1, dtype = torch.int16)   # all-ones words: 0xFF code bytes
        row[0, 0] = sw
        table = row.pin_memory()
        ptr = ext.trellis_embed_register(table)
        try:
            cs = torch.full((D,), scale, dtype = torch.float16)
            ids = torch.tensor([0], dtype = torch.int64)
            ref = dequant_rows_transformed(row, cs, K, seed, D, G, row_ids = ids)
            out = torch.empty(1, D, dtype = torch.float32, device = dev)
            ext.trellis_embed_gather(ptr, mul1_codebook(dev), cs.float().to(dev),
                                     ids.to(dev), K, seed, 1, out)
            assert torch.equal(out.cpu(), ref), \
                f"kernel edge case mismatch (scale {scale}, seed {seed})"
        finally:
            ext.trellis_embed_unregister(table)


@torch.inference_mode()
def test_register_rejects_bad_table():
    """Host-side validation must reject a non-pinned table and a non-contiguous (but
    pinned) table with clean errors; both checks fire before any CUDA call."""
    if not cuda:
        _skip_no_cuda()
        return
    if pytest is None:   # script mode without pytest: pytest.raises unavailable
        return
    from exllamav3.ext import exllamav3_ext as ext
    t = torch.randint(-32768, 32767, (64, 1024), dtype = torch.int16)   # pageable storage
    with pytest.raises(RuntimeError, match = "pinned"):
        ext.trellis_embed_register(t)
    base = torch.randint(-32768, 32767, (64, 1024), dtype = torch.int16).pin_memory()
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.trellis_embed_register(base.t())


def _write_source_table(path: str, key: str, rows: torch.Tensor):
    N, D = rows.shape
    wr = StreamingSafetensorsWriter(path, stream_tensors = [(key, (N, D))],
                                    stream_dtype_str = "F32", small_tensors = {},
                                    metadata = {}, chunk_rows = max(1, N // 4))
    for lo in range(0, N, max(1, N // 4)):
        wr.write_chunk(rows[lo:lo + max(1, N // 4)])
    wr.finalize()


@torch.inference_mode()
def test_converter_module_end_to_end():
    """Synthetic full pipeline: source table -> conversion engine -> .safetensors file ->
    STC scan -> Embedding trellis load -> forward bit-exact vs. the reference codec's
    decode of the same packed bytes; plus the vocab-mismatch assert."""
    if not cuda:
        _skip_no_cuda()
        return
    if _skip_no_avx2():
        return
    from exllamav3.loader.safetensors import SafetensorsCollection as SafeTensors
    from exllamav3.conversion.embed import quantize_trellis_table, TrellisEmbedTableReader
    from exllamav3.modules.embedding import Embedding

    torch.manual_seed(13)
    V, D, K, seed = 192, 5120, 8, 777
    key = "model.embed_tokens.weight"
    rows = torch.randn(V, D) * torch.randn(D).abs().clamp(0.05) + 0.01

    d = tempfile.mkdtemp(prefix = "trellis_e2e_")
    try:
        _write_source_table(os.path.join(d, "src.safetensors"), key, rows)
        out_file = os.path.join(d, "embedding-trellis.safetensors")
        stats = quantize_trellis_table(d, out_file, K = K, seed = seed, key = key,
                                       chunk_rows = 64, devices = [0], verbose = False)
        assert stats["rows"] == V and stats["sqnr_db"] > 30.0

        # module load through the real loader path
        stc = SafeTensors(d, load_method = "python")
        module = Embedding(config = types.SimpleNamespace(stc = stc), key = key.rsplit(".", 1)[0],
                           vocab_size = V, hidden_size = D, out_dtype = torch.float32)
        # production-shaped load: Embedding sets caps["prefer_cpu"], so the loader always
        # loads the module on CPU and passes the compute device separately (model_ls.py) -
        # the registration must land on the compute device, not the load device
        module.load(torch.device("cpu"), compute_device = torch.device(dev))
        assert module.device.type == "cpu"
        assert module.trellis["dev"] == torch.device(dev)
        assert module.trellis is not None and module.trellis["table_ptr"] != 0
        assert module.weights_numel() == V * D

        ids = torch.tensor([[0, 1, 2, V - 1, V // 2, 7, 7, 191]], dtype = torch.int64, device = dev)
        x = module.forward(ids[0], {})
        assert x.shape == (8, D) and x.dtype == torch.float32 and x.device.type == "cuda"

        rd = TrellisEmbedTableReader(out_file)
        packed = rd.read_rows_packed(ids[0].cpu())
        ref = dequant_rows_transformed(packed, rd.col_scales, rd.K, rd.seed, rd.hidden, rd.G,
                                       row_ids = ids[0].cpu())
        rd.close()
        assert torch.equal(x.cpu(), ref), "module forward != reference decode"
        module.unload()
        assert module.trellis is None

        # vocab mismatch must be rejected at load
        bad = Embedding(config = types.SimpleNamespace(stc = stc), key = module.key,
                        vocab_size = V + 1, hidden_size = D, out_dtype = torch.float32)
        try:
            bad.load(torch.device("cpu"), compute_device = torch.device(dev))
            raise AssertionError("vocab mismatch was not asserted")
        except AssertionError as e:
            assert "vocab_size" in str(e)
    finally:
        shutil.rmtree(d, ignore_errors = True)


@torch.inference_mode()
def test_converter_resume_and_keep_rows():
    """Resume on a COMPLETE table is a no-op (re-validates the header parameters, writes
    nothing) and mismatched parameters are rejected; keep_rows re-maps rows to output
    order with scales fitted over the kept set. (Interrupted-run resume with bit-identical
    completion is covered by test_converter_legacy_header_upgrade.)"""
    from exllamav3.conversion.embed import quantize_trellis_table, TrellisEmbedTableReader
    if _skip_no_avx2():
        return

    torch.manual_seed(17)
    V, D, K = 128, 1024, 6
    key = "model.embed_tokens.weight"
    rows = torch.randn(V, D)
    d = tempfile.mkdtemp(prefix = "trellis_resume_")
    try:
        _write_source_table(os.path.join(d, "src.safetensors"), key, rows)

        full = os.path.join(d, "full.safetensors")
        quantize_trellis_table(d, full, K = K, seed = 3, key = key, chunk_rows = 32,
                               devices = [0], verbose = False)
        # resume on the complete file: the writer-level interruption path cannot be
        # simulated here (limit_rows records a SHORT table), so assert the no-op resume
        # and the mismatched-parameter rejection; interrupted resume itself is covered
        # by test_converter_legacy_header_upgrade
        stats = quantize_trellis_table(d, full, K = K, seed = 3, key = key, chunk_rows = 32,
                                       devices = [0], resume = True, verbose = False)
        assert stats["processed_rows"] == 0   # already complete
        try:
            quantize_trellis_table(d, full, K = 7, seed = 3, key = key, chunk_rows = 32,
                                   devices = [0], resume = True, verbose = False)
            raise AssertionError("resume with mismatched K was not rejected")
        except ValueError as e:
            assert "resume" in str(e).lower()

        # keep_rows: output row n = source row keep_rows[n]; scales over kept set
        keep = torch.tensor([5, 0, 127, 64, 64, 3], dtype = torch.int64)
        kp = os.path.join(d, "keep.safetensors")
        quantize_trellis_table(d, kp, K = K, seed = 3, key = key, keep_rows = keep,
                               chunk_rows = 2, devices = [0], verbose = False)
        rd = TrellisEmbedTableReader(kp)
        assert rd.num_rows == keep.shape[0]
        deq = rd.dequant(torch.arange(keep.shape[0]))
        # output row n = source row keep[n]: rows 3 and 4 share source row 64 (distinct
        # sign-stream keys, since the key is the OUTPUT id, so they are encoded differently
        # but must both reconstruct their source within the K = 6 noise floor
        assert ((deq[3] - rows[64]).norm() / rows[64].norm() < 0.03).item()
        assert ((deq[4] - rows[64]).norm() / rows[64].norm() < 0.03).item()
        assert ((deq[0] - rows[5]).norm() / rows[5].norm() < 0.03).item()
        assert ((deq[2] - rows[127]).norm() / rows[127].norm() < 0.03).item()
        rd.close()
    finally:
        shutil.rmtree(d, ignore_errors = True)

@torch.inference_mode()
def test_converter_keep_rows_resume_rejection():
    """Resuming with a DIFFERENT keep_rows row filter must be rejected (the stored
    column scales were fit on the stored run's kept rows): the filter's identity is
    pinned in the file metadata (keep_rows_sha256). The same filter resumes cleanly."""
    from exllamav3.conversion.embed import quantize_trellis_table
    if _skip_no_avx2():
        return
    torch.manual_seed(31)
    V, D, K = 128, 1024, 6
    key = "model.embed_tokens.weight"
    rows = torch.randn(V, D)
    d = tempfile.mkdtemp(prefix = "trellis_keepres_")
    try:
        _write_source_table(os.path.join(d, "src.safetensors"), key, rows)
        out = os.path.join(d, "keep.safetensors")
        keep_a = torch.tensor([0, 1, 2, 3], dtype = torch.int64)
        quantize_trellis_table(d, out, K = K, seed = 3, key = key, keep_rows = keep_a,
                               chunk_rows = 2, devices = [0], verbose = False)
        keep_b = torch.tensor([0, 1, 2, 4], dtype = torch.int64)
        try:
            quantize_trellis_table(d, out, K = K, seed = 3, key = key, keep_rows = keep_b,
                                   chunk_rows = 2, devices = [0], resume = True, verbose = False)
            raise AssertionError("resume with a different keep_rows filter was not rejected")
        except ValueError as e:
            assert "keep_rows" in str(e)
        # same filter: resume is accepted (no-op on the complete file)
        stats = quantize_trellis_table(d, out, K = K, seed = 3, key = key, keep_rows = keep_a,
                                       chunk_rows = 2, devices = [0], resume = True, verbose = False)
        assert stats["processed_rows"] == 0
    finally:
        shutil.rmtree(d, ignore_errors = True)


def test_converter_legacy_header_upgrade():
    """An interrupted run written before keep_rows_sha256 existed (field absent from the
    header) must be upgraded in place on resume - data section untouched - and finish
    byte-identical to a fresh uninterrupted run (determinism: same rows + K/seed)."""
    import json
    import struct
    from exllamav3.conversion.embed import quantize_trellis_table
    if _skip_no_avx2():
        return

    torch.manual_seed(23)
    V, D, K = 128, 1024, 6
    key = "model.embed_tokens.weight"
    rows = torch.randn(V, D)
    d = tempfile.mkdtemp(prefix = "trellis_legacy_")
    try:
        _write_source_table(os.path.join(d, "src.safetensors"), key, rows)

        fresh = os.path.join(d, "fresh.safetensors")
        quantize_trellis_table(d, fresh, K = K, seed = 3, key = key, chunk_rows = 32,
                               devices = [0], verbose = False)
        fresh_bytes = open(fresh, "rb").read()

        # simulate an interrupted pre-field run: strip the field from the header and cut
        # the stream to one complete chunk (the small col_scales tensor stays intact)
        part = os.path.join(d, "part.safetensors")
        quantize_trellis_table(d, part, K = K, seed = 3, key = key, chunk_rows = 32,
                               devices = [0], verbose = False)
        with open(part, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
            data = f.read()
        del header["__metadata__"]["keep_rows_sha256"]
        hj = json.dumps(header, separators = (",", ":")).encode("utf-8")
        hj += b" " * (-len(hj) % 8)
        # Layout coupling to StreamingSafetensorsWriter (conversion/ngram.py, same writer
        # the engine uses): 8-byte LE header length + JSON header padded to a multiple of
        # 8; the data section starts at 8 + hlen with the small tensors first (col_scales
        # = D * 2 bytes) and the stream rows (row_bytes each) behind them. If the writer
        # layout ever changes, this slice must change with it.
        G = D // 256
        row_bytes = (G + D * K // 16) * 2
        with open(part, "wb") as f:
            f.write(struct.pack("<Q", len(hj)))
            f.write(hj)
            f.write(data[: D * 2 + 32 * row_bytes])

        stats = quantize_trellis_table(d, part, K = K, seed = 3, key = key, chunk_rows = 32,
                                       devices = [0], resume = True, verbose = False)
        assert stats["processed_rows"] == V - 32
        resumed_bytes = open(part, "rb").read()
        assert resumed_bytes == fresh_bytes, "resumed legacy run != fresh run"
        hlen = struct.unpack("<Q", resumed_bytes[:8])[0]
        meta = json.loads(resumed_bytes[8:8 + hlen])["__metadata__"]
        assert meta["keep_rows_sha256"] == ""
    finally:
        shutil.rmtree(d, ignore_errors = True)


def _cpu_trellis_module(V: int, D: int, K: int, seed: int):
    """Embedding module with a synthetic trellis table loaded through the real load() path,
    forced onto the CPU fallback branch (dev=None) by mocking torch.cuda.is_available;
    returns (module, packed, col_scales). No safetensors file or conversion engine needed."""
    from exllamav3.modules.embedding import Embedding
    key = "model.embed_tokens"
    _, packed, cs = _synth_table(V, D, K, seed)
    G = D // GROUP
    meta = {"format": et.FORMAT, "version": et.FORMAT_VERSION, "codebook": "mul1",
            "transform": "qtip1", "K": K, "G": G, "seed": seed, "rows": V, "hidden": D}
    tensors = {key + ".weight_trellis": packed, key + ".col_scales": cs}
    stc = types.SimpleNamespace(
        tensor_file_map = {k: "f" for k in tensors},
        file_headers = {"f": {"__metadata__": meta}},
        has_tensor = lambda k: k in tensors,
        get_tensor = lambda k, device, **kw: tensors[k].to(device),
    )
    stc.find_stc = lambda k: stc   # owning-collection hop used by _load_trellis
    module = Embedding(config = types.SimpleNamespace(stc = stc), key = key,
                       vocab_size = V, hidden_size = D, out_dtype = torch.float32)
    with mock.patch("torch.cuda.is_available", return_value = False):
        module.load(torch.device("cpu"))
    assert module.trellis is not None and module.trellis["dev"] is None
    return module, packed, cs


@torch.inference_mode()
def test_module_cpu_fallback_forward():
    """CPU fallback (dev=None): forward must decode rows with the torch reference codec,
    bit-exact with dequant_rows_transformed of the same packed table."""
    if _skip_no_avx2():
        return
    V, D, K, seed = 32, 512, 8, 42
    G = D // GROUP
    module, packed, cs = _cpu_trellis_module(V, D, K, seed)
    try:
        ids = torch.tensor([0, 3, 7, 7, 12, 31])
        x = module.forward(ids, {})
        assert x.shape == (6, D) and x.dtype == torch.float32
        ref = dequant_rows_transformed(packed[ids], cs, K, seed, D, G, row_ids = ids)
        assert torch.equal(x, ref), "CPU fallback forward != reference decode"
    finally:
        module.unload()


@torch.inference_mode()
def test_module_mm_indexed_forward():
    """MM/indexed forward: standard positions (ids < FIRST_MM_EMBEDDING_INDEX) must be
    filled by the trellis _gather and indexed positions by the external embeddings, in
    one combined output."""
    if _skip_no_avx2():
        return
    from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
    V, D, K, seed = 32, 512, 8, 42
    G = D // GROUP
    module, packed, cs = _cpu_trellis_module(V, D, K, seed)
    try:
        mm_emb = torch.randn(2, D)
        ie = types.SimpleNamespace(first_index = FIRST_MM_EMBEDDING_INDEX, mm_length = 2,
                                   deepstack_embeddings = None, embeddings = mm_emb)
        mm0 = FIRST_MM_EMBEDDING_INDEX
        ids = torch.tensor([[0, 1, mm0, mm0 + 1, 3],
                            [5, mm0 + 1, 9, 9, 31]])
        x = module.forward(ids, {"indexed_embeddings": [ie]})
        assert x.shape == (2, 5, D) and x.dtype == torch.float32
        for i in range(2):
            sm = ids[i] < mm0
            im = (ids[i] >= mm0) & (ids[i] < mm0 + 2)
            assert bool(sm.any()) and bool(im.any())
            std_ids = ids[i][sm]
            exp = dequant_rows_transformed(packed[std_ids], cs, K, seed, D, G, row_ids = std_ids)
            assert torch.equal(x[i][sm], exp), f"MM forward row {i}: standard gather mismatch"
            assert torch.equal(x[i][im], mm_emb[ids[i][im] - mm0]), \
                f"MM forward row {i}: indexed fill mismatch"
    finally:
        module.unload()


@torch.inference_mode()
def test_module_oob_id_contract():
    """Current (deliberately relaxed) OOB-id contract: a loud IndexError only when the
    ids are HOST-resident (checked before the H2D copy - both the CPU fallback and the
    CUDA branch); ids that arrive already on the compute device skip the check and the
    kernel clamps them to the last row (bit-identical to row N-1). The kernel-level
    clamp itself is pinned by test_kernel_oob_id_clamped."""
    if not cuda:
        _skip_no_cuda()
        return
    if _skip_no_avx2():
        return
    from exllamav3.loader.safetensors import SafetensorsCollection as SafeTensors
    from exllamav3.conversion.embed import quantize_trellis_table
    from exllamav3.modules.embedding import Embedding

    torch.manual_seed(29)
    V, D, K, seed = 64, 512, 8, 5
    key = "model.embed_tokens.weight"
    rows = torch.randn(V, D)
    d = tempfile.mkdtemp(prefix = "trellis_oob_")
    try:
        _write_source_table(os.path.join(d, "src.safetensors"), key, rows)
        out_file = os.path.join(d, "embedding-trellis.safetensors")
        quantize_trellis_table(d, out_file, K = K, seed = seed, key = key,
                               chunk_rows = 16, devices = [0], verbose = False)
        stc = SafeTensors(d, load_method = "python")
        module = Embedding(config = types.SimpleNamespace(stc = stc), key = key.rsplit(".", 1)[0],
                           vocab_size = V, hidden_size = D, out_dtype = torch.float32)
        module.load(torch.device("cpu"), compute_device = torch.device(dev))
        try:
            # host-resident OOB id: loud IndexError on the CUDA branch (before the H2D copy)
            if pytest is not None:
                with pytest.raises(IndexError, match = "out of range"):
                    module.forward(torch.tensor([V]), {})
            # already-on-device OOB id: no check, the kernel clamps to the last row
            oob = module.forward(torch.tensor([V], device = dev), {})
            ref = module.forward(torch.tensor([V - 1], device = dev), {})
            assert torch.equal(oob, ref), "cuda-resident OOB id must clamp to the last row"
        finally:
            module.unload()

        # CPU fallback branch: host-resident OOB id is loud as well
        module2, _, _ = _cpu_trellis_module(32, 512, 8, 42)
        try:
            if pytest is not None:
                with pytest.raises(IndexError, match = "out of range"):
                    module2.forward(torch.tensor([32]), {})
        finally:
            module2.unload()
    finally:
        shutil.rmtree(d, ignore_errors = True)


@torch.inference_mode()
def test_module_retarget_trellis():
    """retarget_trellis: a same-device retarget is a no-op (stable table_ptr, dev
    unchanged); with >= 2 GPUs the registration moves to the new device and the gather
    stays bit-exact with the reference decode there."""
    if not cuda:
        _skip_no_cuda()
        return
    if _skip_no_avx2():
        return
    from exllamav3.loader.safetensors import SafetensorsCollection as SafeTensors
    from exllamav3.conversion.embed import quantize_trellis_table
    from exllamav3.modules.embedding import Embedding

    torch.manual_seed(37)
    V, D, K, seed = 64, 512, 8, 11
    key = "model.embed_tokens.weight"
    rows = torch.randn(V, D)
    d = tempfile.mkdtemp(prefix = "trellis_retarget_")
    try:
        _write_source_table(os.path.join(d, "src.safetensors"), key, rows)
        out_file = os.path.join(d, "embedding-trellis.safetensors")
        quantize_trellis_table(d, out_file, K = K, seed = seed, key = key,
                               chunk_rows = 16, devices = [0], verbose = False)
        stc = SafeTensors(d, load_method = "python")
        module = Embedding(config = types.SimpleNamespace(stc = stc), key = key.rsplit(".", 1)[0],
                           vocab_size = V, hidden_size = D, out_dtype = torch.float32)
        module.load(torch.device("cpu"), compute_device = torch.device(dev))
        try:
            # same device: no-op, stable alias
            p0 = module.trellis["table_ptr"]
            module.retarget_trellis(torch.device(dev))
            assert module.trellis["table_ptr"] == p0
            assert module.trellis["dev"] == torch.device(dev)
            if torch.cuda.device_count() >= 2:
                dev2 = torch.device("cuda:1")
                module.retarget_trellis(dev2)
                assert module.trellis["dev"] == dev2
                assert module.trellis["table_ptr"] != p0
                ids = torch.tensor([0, 1, V - 1], dtype = torch.int64)
                x = module.forward(ids, {})
                assert x.device == dev2
                ref = dequant_rows_transformed(module.trellis["table"][ids],
                                               module.trellis["col_scales"], K, seed, D,
                                               D // GROUP, row_ids = ids)
                assert torch.equal(x.cpu(), ref), "retargeted gather != reference decode"
        finally:
            module.unload()
    finally:
        shutil.rmtree(d, ignore_errors = True)


@torch.inference_mode()
def test_tp_export_rejects_trellis():
    """tp_export must refuse a trellis-quantized embedding (no TP plan exists for the
    device-mapped host table)."""
    from exllamav3.modules.embedding import Embedding
    module = Embedding(config = None, key = "model.embed_tokens", vocab_size = 32,
                       hidden_size = 512)
    module.device = torch.device("cpu")
    module.trellis = {"table_ptr": None}
    try:
        module.tp_export(None, None)
        raise AssertionError("tp_export did not reject a trellis embedding")
    except AssertionError as e:
        assert "tensor parallelism" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print(f"OK {f.__name__}")
    print("all embed_trellis tests passed")
