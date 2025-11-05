import ginfer_test
import numpy as np
import pytest

@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
def test_add_layer_cuda(dtype):
    a = np.random.rand(3, 127, 127).astype(dtype)
    b = np.random.rand(3, 127, 127).astype(dtype)
    out = ginfer_test.run_add_layer_cuda_test(a, b)
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    atol = 1e-1 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, a + b, rtol=rtol, atol=atol)

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_rmsnorm_layer_cuda(dtype):
    input = np.random.rand(3, 127, 127).astype(dtype)
    gamma = np.random.rand(127).astype(dtype)
    epsilon = 1e-5
    out = ginfer_test.run_rmsnorm_layer_cuda_test(input, gamma, epsilon)
    ref = input * gamma / np.sqrt(np.mean(input**2, axis=-1, keepdims=True) + epsilon)
    atol = 1e-3 if dtype == np.float16 else 1e-5
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)