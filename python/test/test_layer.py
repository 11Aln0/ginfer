import ginfer_test
import numpy as np
import pytest

@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
def test_add_layer_cuda(dtype):
    a = np.random.rand(3, 127, 127).astype(dtype)
    b = np.random.rand(3, 127, 127).astype(dtype)
    out = ginfer_test.test_add_layer_cuda(a, b)
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    atol = 1e-1 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, a + b, rtol=rtol, atol=atol)

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_rmsnorm_layer_cuda(dtype):
    input = np.random.rand(3, 127, 127).astype(dtype)
    gamma = np.random.rand(127).astype(dtype)
    epsilon = 1e-5
    out = ginfer_test.test_rmsnorm_layer_cuda(input, gamma, epsilon)
    ref = input * gamma / np.sqrt(np.mean(input**2, axis=-1, keepdims=True) + epsilon)
    atol = 1e-3 if dtype == np.float16 else 1e-5
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)
    

@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_matmul_layer_gemv_cuda(dtype):
    a = np.random.rand(64).astype(dtype)
    b = np.random.rand(64, 64).astype(dtype, order='F') # col-major
    out = ginfer_test.test_matmul_layer_cuda(a, b)
    ref = np.matmul(a, b)
    atol = 1e-3 if dtype == np.float16 else 1e-5
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)