import ginfer_test
import numpy as np
import pytest
import torch
import torch.nn.functional as F

@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
def test_add_layer_cuda(dtype):
    a = np.random.rand(3, 127, 128).astype(dtype)
    b = np.random.rand(3, 127, 128).astype(dtype)
    out = ginfer_test.test_add_layer_cuda(a, b)
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    atol = 1e-1 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, a + b, rtol=rtol, atol=atol)

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_rmsnorm_layer_cuda(dtype):
    input = np.random.rand(3, 127, 128).astype(dtype)
    gamma = np.random.rand(128).astype(dtype)
    epsilon = 1e-5
    out = ginfer_test.test_rmsnorm_layer_cuda(input, gamma, epsilon)
    ref = input * gamma / np.sqrt(np.mean(input**2, axis=-1, keepdims=True) + epsilon)
    atol = 1e-3 if dtype == np.float16 else 1e-5
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)
    
@pytest.mark.parametrize("k, n", [(128, 1000), (512, 256), (4216, 197)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_matmul_layer_gemv_cuda(dtype, k, n):
    a = np.random.randn(k).astype(dtype)
    b = np.random.randn(k, n).astype(dtype, order='F') # col-major
    out = ginfer_test.test_matmul_layer_cuda(a, b)
    ref = np.matmul(a, b)
    atol = 2e-2 if dtype == np.float16 else 5e-5
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)

GEMM_CONFIG = [
    # (2, 8, 9),
    (1031, 2000, 408),
    (512, 1024, 256),
    (4090, 1000, 1000)
]

@pytest.mark.parametrize("dtype", [np.float16])
@pytest.mark.parametrize("m,n,k", GEMM_CONFIG)
def test_matmul_layer_gemm_cuda(dtype, m, n, k):
    # currently n/k must be 16 bytes alignment
    a = np.random.randn(m, k).astype(dtype)
    b = np.random.randn(k, n).astype(dtype, order='F') # col-major
    out = ginfer_test.test_matmul_layer_cuda(a, b)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a_t = torch.from_numpy(a).to(device)
    b_t = torch.from_numpy(b).to(device)
    ref = torch.matmul(a_t, b_t).cpu().numpy()
    atol = 3e-2 if dtype == np.float16 else 1e-5 # TODO more accurate 
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)


def torch_gqa_sdpa_reference(q_np, k_np, v_np, seq_len, is_causal=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 注意：为了做 Golden Check，使用 float32 避免累积误差
    q = torch.from_numpy(q_np).to(device).float()
    k = torch.from_numpy(k_np).to(device).float()
    v = torch.from_numpy(v_np).to(device).float()

    q = q[:, :seq_len, :, :]
    k = k[:, :seq_len, :, :]
    v = v[:, :seq_len, :, :]

    # [B, L, H, D] -> [B, H, L, D]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    B, num_heads, L, head_dim = q.shape
    kv_heads = k.shape[1]

    # GQA 广播逻辑
    if num_heads != kv_heads:
        n_rep = num_heads // kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    # [B, H, L, D] -> [B, L, H, D]
    out = out.permute(0, 2, 1, 3)

    return out.detach().cpu().numpy()

MODEL_CONFIGS = [
    ("Llama3-8B", 32, 8, 128, 8192),
    ("GPT2-Base", 12, 12, 64, 1024),
    ("BERT-Base", 12, 12, 64, 512),
]

@pytest.mark.parametrize("name, num_heads, kv_heads, head_dim, max_seq_len", MODEL_CONFIGS)
@pytest.mark.parametrize("seq_len", [127, 258, 500, 1125, 2000, 5000])
@pytest.mark.parametrize("dtype", [np.float16])
def test_attention_layer_suite(name, num_heads, kv_heads, head_dim, max_seq_len, seq_len, dtype):
    
    seq_len = min(max_seq_len, seq_len) 
    batch_size = 2
    
    # 遵循 [B, MaxL, H, D] 布局
    q = np.random.randn(batch_size, max_seq_len, num_heads, head_dim).astype(dtype)
    k = np.random.randn(batch_size, max_seq_len, kv_heads, head_dim).astype(dtype)
    v = np.random.randn(batch_size, max_seq_len, kv_heads, head_dim).astype(dtype)

    out_cuda = ginfer_test.test_gqa_layer_cuda(q, k, v, seq_len)
    
    # 提取有效区域 [B, seq_len, H, D]
    out_cuda_valid = out_cuda[:, :seq_len, :, :]

    # PyTorch Reference
    ref_out = torch_gqa_sdpa_reference(q, k, v, seq_len, is_causal=True)


    np.testing.assert_allclose(
        out_cuda_valid.astype(np.float32), 
        ref_out.astype(np.float32), 
        rtol=5e-3, 
        atol=5e-3,
        err_msg=f"Logic error in {name} config!"
    )
    
# ==================== Argmax ====================

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_argmax_layer_cuda(dtype):
    data = np.random.randn(4096).astype(dtype)
    out = ginfer_test.test_argmax_layer_cuda(data)
    idx = int(out[0])
    expected = int(np.argmax(data))
    assert idx == expected


# ==================== Embedding ====================

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_embedding_layer_cuda(dtype):
    vocab_size = 1000
    embedding_dim = 128
    seq_len = 32
    indices = np.random.randint(0, vocab_size, size=(seq_len,)).astype(np.int64)
    weight = np.random.randn(vocab_size, embedding_dim).astype(dtype)
    out = ginfer_test.test_embedding_layer_cuda(indices, weight)
    ref = weight[indices]
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    atol = 1e-3 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)


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
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_rope_layer_cuda(dtype, seq_len, nhead, head_dim, rope_theta):
    x = np.random.randn(seq_len, nhead, head_dim).astype(dtype)
    start_pos = 0
    end_pos = seq_len
    out = ginfer_test.test_rope_layer_cuda(x, head_dim, start_pos, end_pos, rope_theta)
    ref = rope_reference(x, start_pos, rope_theta).astype(dtype)
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


# ==================== SwiGLU ====================

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_swiglu_layer_cuda(dtype):
    gate = np.random.randn(4, 128, 256).astype(dtype)
    up = np.random.randn(4, 128, 256).astype(dtype)
    out = ginfer_test.test_swiglu_layer_cuda(gate, up)
    # SwiGLU(gate, up) = silu(gate) * up = (gate * sigmoid(gate)) * up
    gate_f = gate.astype(np.float32)
    up_f = up.astype(np.float32)
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_f))
    ref = gate_f * sigmoid_gate * up_f
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    atol = 1e-2 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=rtol, atol=atol)