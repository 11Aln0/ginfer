import ginfer_test
import numpy as np
import ml_dtypes
import pytest
import torch
import torch.nn.functional as F

# ==================== Add ====================

@pytest.mark.parametrize("dtype, atol, rtol", [
    (np.float32, 1e-5, 1e-5),
    (np.float16, 1e-3, 1e-3),
    (np.int32, 0, 0),
    (ml_dtypes.bfloat16, 1e-3, 1e-3)
])
def test_add_op_cuda(dtype, atol, rtol):

    a = np.random.rand(3, 127, 128).astype(dtype)
    b = np.random.rand(3, 127, 128).astype(dtype)
    out = ginfer_test.test_add_op_cuda(a, b)
    np.testing.assert_allclose(out, a + b, rtol=rtol, atol=atol)

# ==================== RMSNorm ====================

@pytest.mark.parametrize("dtype, atol, rtol", [
    (np.float32, 1e-5, 1e-5),
    (np.float16, 1e-3, 1e-3),
    (ml_dtypes.bfloat16, 1e-2, 1e-2)
])
def test_rmsnorm_op_cuda(dtype, atol, rtol):
    input = np.random.rand(3, 127, 128).astype(dtype)
    gamma = np.random.rand(128).astype(dtype)
    epsilon = 1e-5
    out = ginfer_test.test_rmsnorm_op_cuda(input, gamma, epsilon)
    input_f = input.astype(np.float32)
    gamma_f = gamma.astype(np.float32)
    ref = input_f * gamma_f / np.sqrt(np.mean(input_f**2, axis=-1, keepdims=True) + epsilon)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=rtol, atol=atol)

# ==================== GEMV ====================

@pytest.mark.parametrize("k, n", [(128, 1000), (512, 256), (4216, 197)])
@pytest.mark.parametrize("dtype, atol, rtol", [
    (np.float32, 5e-5, 5e-5), # kernel 累加顺序不同，float32 误差略大
    (np.float16, 1e-3, 1e-3),
    (ml_dtypes.bfloat16, 1e-2, 1e-2)
])
def test_matmul_op_gemv_cuda(dtype, atol, rtol, k, n):
    a = np.random.randn(k).astype(dtype)
    b = np.random.randn(k, n).astype(dtype, order='F') # col-major
    out = ginfer_test.test_matmul_op_cuda(a, b)
    ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=rtol, atol=atol)
    
    bias = np.random.randn(n).astype(dtype)
    out_with_bias = ginfer_test.test_matmul_op_with_bias_cuda(a, b, bias)
    ref_with_bias = ref + bias.astype(np.float32)
    np.testing.assert_allclose(out_with_bias.astype(np.float32), ref_with_bias, rtol=rtol, atol=atol)

# ==================== GEMM ====================

GEMM_CONFIG = [
    # (2, 8, 9),
    (1031, 2000, 408),
    (512, 1024, 256),
    (4090, 1000, 1000)
]

@pytest.mark.parametrize("dtype, atol, rtol", [
    (np.float16, 1e-3, 1e-3),
    (ml_dtypes.bfloat16, 1e-2, 1e-2)
])
@pytest.mark.parametrize("m,n,k", GEMM_CONFIG)
def test_matmul_op_gemm_cuda(dtype, atol, rtol, m, n, k):
    # currently n/k must be 16 bytes alignment
    a = np.random.randn(m, k).astype(dtype)
    b = np.random.randn(k, n).astype(dtype, order='F') # col-major
    out = ginfer_test.test_matmul_op_cuda(a, b)
    ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=rtol, atol=atol)
    
    bias =  np.random.randn(n).astype(dtype)
    out_with_bias = ginfer_test.test_matmul_op_with_bias_cuda(a, b, bias)
    ref_with_bias = ref + bias.astype(np.float32)
    np.testing.assert_allclose(out_with_bias.astype(np.float32), ref_with_bias, rtol=rtol, atol=atol)

# ==================== GQA ====================

def torch_gqa_sdpa_reference(q_np, k_np, v_np, is_causal=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 注意：为了做 Golden Check，使用 float32 避免累积误差
    q = torch.from_numpy(q_np.astype(np.float32)).to(device)
    k = torch.from_numpy(k_np.astype(np.float32)).to(device)
    v = torch.from_numpy(v_np.astype(np.float32)).to(device)

    # [L, H, D] -> [H, L, D]
    q = q.permute(1, 0, 2)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)

    num_heads, L, head_dim = q.shape
    S = k.shape[1]
    kv_heads = k.shape[0]

    # GQA 广播逻辑
    if num_heads != kv_heads:
        n_rep = num_heads // kv_heads
        k = k.repeat_interleave(n_rep, dim=0)
        v = v.repeat_interleave(n_rep, dim=0)
    
    if is_causal and L != S:
        # 手动构造 lower-right causal mask
        attn_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=S - L)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    # [H, L, D] -> [L, H, D]
    out = out.permute(1, 0, 2)

    return out.detach().cpu().numpy()

MODEL_CONFIGS = [
    ("Llama3-8B", 32, 8, 128),
    ("GPT2-Base", 12, 12, 64),
    ("Qwen2-1.5B", 12, 2, 128),
]

@pytest.mark.parametrize("name, num_heads, kv_heads, head_dim", MODEL_CONFIGS)
@pytest.mark.parametrize("q_seq_len, kv_seq_len", [(127, 127), (1, 258), (9, 9), (1125, 1600), (1, 2000), (2500, 5000)])
@pytest.mark.parametrize("dtype, atol, rtol", [
    (np.float16, 2e-3, 2e-3),
    (ml_dtypes.bfloat16, 2e-2, 2e-2)
])
def test_gqa_op(dtype, atol, rtol, name, num_heads, kv_heads, head_dim, q_seq_len, kv_seq_len):

    # batch_size = 2

    # 遵循 [B, MaxL, H, D] 布局
    q = np.random.randn(q_seq_len, num_heads, head_dim).astype(dtype)
    k = np.random.randn(kv_seq_len, kv_heads, head_dim).astype(dtype)
    v = np.random.randn(kv_seq_len, kv_heads, head_dim).astype(dtype)

    out_cuda = ginfer_test.test_gqa_op_cuda(q, k, v)

    # PyTorch Reference
    ref_out = torch_gqa_sdpa_reference(q, k, v, is_causal=True)

    out_cuda = out_cuda.astype(np.float32)
    if not np.allclose(out_cuda, ref_out, rtol=rtol, atol=atol):
        print("Max absolute error:", np.max(np.abs(out_cuda - ref_out)))
        print("Max relative error:", np.max(np.abs(out_cuda - ref_out) / (np.abs(ref_out) + 1e-6)))
    np.testing.assert_allclose(out_cuda, ref_out, rtol=rtol, atol=atol)


# ==================== Argmax ====================

@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("tensor_shape", [(128,), (1024,), (4096,), (2, 128), (17, 1024), (33, 4096)])
def test_argmax_op_cuda(dtype, tensor_shape):
    data = np.random.randn(*tensor_shape).astype(dtype)
    out = ginfer_test.test_argmax_op_cuda(data)
    expected = np.argmax(data, axis=-1).astype(np.int64) if data.ndim == 2 else np.array([np.argmax(data)], dtype=np.int64)
    
    np.testing.assert_array_equal(out, expected)


# ==================== Embedding ====================

@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("seq_len, vocab_size, embedding_dim", [
    (32, 1000, 128),
    (64, 5000, 256),
    (127, 10000, 512)
])
def test_embedding_op_cuda(dtype, seq_len, vocab_size, embedding_dim):
    indices = np.random.randint(0, vocab_size, size=(seq_len,)).astype(np.int32)
    weight = np.random.randn(vocab_size, embedding_dim).astype(dtype)
    out = ginfer_test.test_embedding_op_cuda(indices, weight)
    ref = weight[indices]
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


# ==================== ROPE ====================

def rope_reference(x, start_pos, rope_theta=10000.0):
    """Compute RoPE (Rotary Position Embedding) reference."""
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    orig_shape = x.shape
    # input layout: [seq_len, nhead, head_dim]
    x = x.reshape(-1, x.shape[-2], x.shape[-1])
    seq_len = x.shape[0]

    positions = np.arange(start_pos, start_pos + seq_len, dtype=np.float32)
    freqs = np.array([1.0 / (rope_theta ** (2.0 * i / head_dim)) for i in range(half_dim)], dtype=np.float32)
    theta = np.outer(positions, freqs)  # [seq_len, half_dim]
    sin_vals = np.sin(theta)[:, np.newaxis, :]  # [seq_len, 1, half_dim]
    cos_vals = np.cos(theta)[:, np.newaxis, :]

    x_even = x[..., :half_dim]
    x_odd = x[..., half_dim:]

    out = np.empty_like(x)
    out[..., :half_dim] = x_even * cos_vals - x_odd * sin_vals
    out[..., half_dim:] = x_even * sin_vals + x_odd * cos_vals
    return out.reshape(orig_shape)


ROPE_CONFIGS = [
    # (seq_len, nhead, head_dim, rope_theta)
    (128, 32, 128, 10000.0),  # Llama-style
    (64, 12, 64, 10000.0),    # GPT2-style
]

@pytest.mark.parametrize("seq_len, nhead, head_dim, rope_theta", ROPE_CONFIGS)
@pytest.mark.parametrize("dtype, atol, rtol", [
    (np.float32, 1e-4, 1e-4),
    (np.float16, 1e-3, 1e-3),
    (ml_dtypes.bfloat16, 1e-2, 1e-2)
])
def test_rope_op_cuda(dtype, atol, rtol, seq_len, nhead, head_dim, rope_theta):
    x = np.random.randn(seq_len, nhead, head_dim).astype(dtype)
    start_pos = 0
    end_pos = seq_len - 1
    out = ginfer_test.test_rope_op_cuda(x, head_dim, start_pos, end_pos, rope_theta)
    ref = rope_reference(x, start_pos, rope_theta).astype(dtype)
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)


# ==================== SwiGLU ====================

@pytest.mark.parametrize("dtype, atol, rtol", [
    (np.float32, 1e-5, 1e-5),
    (np.float16, 1e-3, 1e-3),
    (ml_dtypes.bfloat16, 1e-2, 1e-2)
])
@pytest.mark.parametrize("batch_size, seq_len, hidden_dim", [
    (4, 128, 256),
    (2, 511, 1024),
    (1, 1023, 2048)
])
def test_swiglu_op_cuda(dtype, atol, rtol, batch_size, seq_len, hidden_dim):
    gate = np.random.randn(batch_size, seq_len, hidden_dim).astype(dtype)
    up = np.random.randn(batch_size, seq_len, hidden_dim).astype(dtype)
    out = ginfer_test.test_swiglu_op_cuda(gate, up)
    # SwiGLU(gate, up) = silu(gate) * up = (gate * sigmoid(gate)) * up
    gate_f = gate.astype(np.float32)
    up_f = up.astype(np.float32)
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_f))
    ref = gate_f * sigmoid_gate * up_f
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=rtol, atol=atol)
