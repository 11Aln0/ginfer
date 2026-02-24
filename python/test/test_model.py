import ginfer_test
import numpy as np
import pytest
import ml_dtypes
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download

# ==================== Qwen2 ====================

MODEL_PATH = os.environ.get("QWEN2_MODEL_PATH", "")

def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_name,
        trust_remote_code=True,
    )

    return model, tokenizer, model_path


def hf_generate(
    inputs, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    return outputs[0]

def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    output = hf_generate(inputs, model, max_new_tokens, top_p, top_k, temperature)
    result = tokenizer.decode(output, skip_special_tokens=True)
    return result

def test_qwen2_generate_cuda():
    # <｜User｜>Who are you?<｜Assistant｜><think>
    input_ids = np.array([151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198], dtype=np.int32)
    seq_len = len(input_ids)
    pos_id_range = (0, seq_len - 1) # closed interval
    
    hf_model, _, model_path = load_hf_model(MODEL_PATH, device_name="cuda:0")
    ref_token_ids = hf_generate(torch.tensor([input_ids], device="cuda:0"), hf_model, max_new_tokens=128, top_p = 1.0, top_k = 1, temperature = 1.0).tolist()
    
    next_token_ids = ginfer_test.test_qwen2_generate_cuda(model_path, input_ids, pos_id_range)
    token_ids = np.concatenate([input_ids, next_token_ids])
    

    np.testing.assert_array_equal(token_ids, ref_token_ids)

def test_qwen2_infer_cuda():
    prompt = "Who are you?"
    hf_model, hf_tokenizer, model_path = load_hf_model(MODEL_PATH, device_name="cuda:1")
    ref_result = hf_infer(prompt, hf_tokenizer, hf_model, max_new_tokens=128, top_p = 1.0, top_k = 1, temperature = 1.0)
    
    result = ginfer_test.test_qwen2_infer_cuda(model_path, prompt)
    
    assert result == ref_result
    