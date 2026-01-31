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
    

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_matmul_layer_gemv_cuda(dtype):
    a = np.random.rand(1024).astype(dtype)
    b = np.random.rand(1024, 1000).astype(dtype, order='F') # col-major
    out = ginfer_test.test_matmul_layer_cuda(a, b)
    ref = np.matmul(a, b)
    atol = 1e-3 if dtype == np.float16 else 1e-5
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
    b = np.random.rand(k, n).astype(dtype, order='F') # col-major
    out = ginfer_test.test_matmul_layer_cuda(a, b)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a_t = torch.from_numpy(a).to(device)
    b_t = torch.from_numpy(b).to(device)
    ref = torch.matmul(a_t, b_t).cpu().numpy()
    atol = 2e-2 if dtype == np.float16 else 1e-5 # TODO more accurate 
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