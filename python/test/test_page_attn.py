import ginfer_test
import numpy as np
import ml_dtypes
import pytest
import torch
from flash_attn import flash_attn_varlen_func

# ==================== GQA Varlen ====================

def flash_attn_varlen_reference(q_cat, k_cache, v_cache, block_tables,
                                cu_seqlens_q, cu_seqlens_kv,
                                max_seqlen_q, max_seqlen_kv, dtype):
    """
    用 flash_attn_varlen_func (with block_table) 计算 causal GQA 参考输出。
    返回: [total_q, nheads, head_dim] numpy array
    """
    device = "cuda"
    torch_dtype = torch.float16 if dtype == np.float16 else torch.bfloat16

    out = flash_attn_varlen_func(
        torch.from_numpy(q_cat.astype(np.float32)).to(device=device, dtype=torch_dtype),
        torch.from_numpy(k_cache.astype(np.float32)).to(device=device, dtype=torch_dtype),
        torch.from_numpy(v_cache.astype(np.float32)).to(device=device, dtype=torch_dtype),
        cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(cu_seqlens_kv, dtype=torch.int32, device=device),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_kv,
        causal=True,
        block_table=torch.from_numpy(block_tables).to(device=device),
    )  # [total_q, nheads, head_dim]

    return out.cpu().to(dtype=torch.float32).numpy()


VARLEN_CONFIGS = [
    # (name, num_heads, kv_heads, head_dim)
    ("Llama3-8B",  32,  8, 128),
    ("Qwen2-1.5B", 12,  2, 128),
    ("GPT2-Base",  12, 12,  64),
]

VARLEN_SEQ_CONFIGS = [
    # [(q_len_0, kv_len_0), (q_len_1, kv_len_1)]
    [(1, 128), (1, 256)],
    [(64, 64), (128, 128)],
    [(1, 512), (9, 9)],
    [(125, 260), (63, 63)],
]


@pytest.mark.parametrize("name, num_heads, kv_heads, head_dim", VARLEN_CONFIGS)
@pytest.mark.parametrize("seq_config", VARLEN_SEQ_CONFIGS)
@pytest.mark.parametrize("dtype, atol, rtol", [
    (np.float16,      2e-3, 2e-3),
    (ml_dtypes.bfloat16, 2e-2, 2e-2),
])
def test_gqa_varlen_op(dtype, atol, rtol, name, num_heads, kv_heads, head_dim, seq_config):
    import math
    paged_block_size = 256  # 与 kernel 中 bc=64 对齐

    q_lens  = [cfg[0] for cfg in seq_config]
    kv_lens = [cfg[1] for cfg in seq_config]
    batch_size = len(seq_config)

    # 预分配 KV cache（物理块空间）
    blocks_per_seq = [math.ceil(l / paged_block_size) for l in kv_lens]
    total_blocks   = sum(blocks_per_seq)
    max_blocks     = max(blocks_per_seq)
    k_cache = np.random.randn(total_blocks, paged_block_size, kv_heads, head_dim).astype(dtype)
    v_cache = np.random.randn(total_blocks, paged_block_size, kv_heads, head_dim).astype(dtype)

    # block_tables: 每条序列顺序占用物理块
    block_tables = np.zeros((batch_size, max_blocks), dtype=np.int32)
    blk_offset = 0
    for i, n in enumerate(blocks_per_seq):
        block_tables[i, :n] = np.arange(blk_offset, blk_offset + n)
        blk_offset += n

    # varlen 辅助张量
    cu_seqlens_q  = np.array([0] + list(np.cumsum(q_lens)),  dtype=np.int32)
    cu_seqlens_kv = np.array([0] + list(np.cumsum(kv_lens)), dtype=np.int32)
    max_seqlen_q  = max(q_lens)
    max_seqlen_kv = max(kv_lens)

    # Q
    q_cat = np.random.randn(sum(q_lens), num_heads, head_dim).astype(dtype)

    # 调用 CUDA kernel
    out_cuda = ginfer_test.test_gqa_varlen_op_cuda(
        q_cat, k_cache, v_cache,
        cu_seqlens_q, cu_seqlens_kv, block_tables,
        max_seqlen_q, paged_block_size,
    ).astype(np.float32)

    # flash_attn 参考（block_table 与 ginfer kernel 完全对等）
    ref_out = flash_attn_varlen_reference(
        q_cat, k_cache, v_cache, block_tables,
        cu_seqlens_q, cu_seqlens_kv,
        max_seqlen_q, max_seqlen_kv, dtype,
    )

    if not np.allclose(out_cuda, ref_out, rtol=rtol, atol=atol):
        print("Max absolute error:", np.max(np.abs(out_cuda - ref_out)))
        print("Max relative error:", np.max(np.abs(out_cuda - ref_out) / (np.abs(ref_out) + 1e-6)))
    np.testing.assert_allclose(out_cuda, ref_out, rtol=rtol, atol=atol)


# ==================== Store KV Cache ====================

STORE_KVCACHE_CONFIGS = [
    # (kv_heads, head_dim)
    (8, 128),
    (2, 128),
    (4,  64),
]

@pytest.mark.parametrize("kv_heads, head_dim", STORE_KVCACHE_CONFIGS)
@pytest.mark.parametrize("dtype, atol", [
    (np.float16,         1e-5),
    (ml_dtypes.bfloat16, 1e-5),
])
def test_store_kvcache_op(dtype, atol, kv_heads, head_dim):
    total_slots = 64   # physical cache pool size
    num_tokens  = 16   # tokens to write this step

    slot_mapping = np.random.choice(total_slots, num_tokens, replace=False).astype(np.int32)

    k = np.random.randn(num_tokens, kv_heads, head_dim).astype(dtype)
    v = np.random.randn(num_tokens, kv_heads, head_dim).astype(dtype)

    # Pre-fill cache with sentinel so untouched slots can be verified
    k_cache = np.full((total_slots, kv_heads, head_dim), fill_value=99.0, dtype=dtype)
    v_cache = np.full((total_slots, kv_heads, head_dim), fill_value=99.0, dtype=dtype)

    # Reference: manual scatter
    ref_k_cache = k_cache.copy()
    ref_v_cache = v_cache.copy()
    for i, slot in enumerate(slot_mapping):
        ref_k_cache[slot] = k[i]
        ref_v_cache[slot] = v[i]

    # CUDA kernel — results written back in-place into k_cache / v_cache numpy arrays
    ginfer_test.test_store_kvcache_op_cuda(k, v, k_cache, v_cache, slot_mapping)

    np.testing.assert_allclose(k_cache.astype(np.float32),
                               ref_k_cache.astype(np.float32), atol=atol)
    np.testing.assert_allclose(v_cache.astype(np.float32),
                               ref_v_cache.astype(np.float32), atol=atol)
