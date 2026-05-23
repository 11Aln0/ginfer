import ginfer_test
import numpy as np
import pytest
import ml_dtypes
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


MODEL_PATH = os.environ.get("MODEL_PATH", "")

def load_hf_model(model_path=None, device_name="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_name,
        trust_remote_code=True,
    )

    return model, tokenizer


def hf_generate(
    inputs, model, max_new_tokens=128,
):
    compiled_model = torch.compile(model, mode="reduce-overhead")
    
    with torch.no_grad():
        # warmup
        compiled_model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            do_sample=False
        )
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        outputs = compiled_model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            do_sample=False
        )
        
        torch.cuda.synchronize()
        print(f"torch compiled generate time: {time.perf_counter() - start:.3f}s")

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

def test_model_generate_cuda():
    # <｜User｜>How are you?<｜Assistant｜><think>
    input_ids = np.array([151646, 151646, 151644, 4340, 525, 498, 30, 151645, 151648, 198], dtype=np.int32)
    
    hf_model, _, = load_hf_model(MODEL_PATH, device_name="cuda:0")
    ref_token_ids = hf_generate(torch.tensor([input_ids], device="cuda:0"), hf_model, max_new_tokens=1024).tolist()
    
    next_token_ids = ginfer_test.test_model_generate_cuda(MODEL_PATH, input_ids)
    token_ids = np.concatenate([input_ids, next_token_ids])
    
    np.testing.assert_array_equal(token_ids, ref_token_ids)

def test_model_infer_cuda():
    prompt = "13.8 and 13.11, which one is larger?"
    hf_model, hf_tokenizer = load_hf_model(MODEL_PATH, device_name="cuda:0")
    ref_result = hf_infer(prompt, hf_tokenizer, hf_model, max_new_tokens=1024)
    
    result = ginfer_test.test_model_infer_cuda(MODEL_PATH, prompt)
    
    print("result", result)
    print("ref_result", ref_result)
    # assert result == ref_result
    