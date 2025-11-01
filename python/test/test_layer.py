import ginfer_test
import numpy as np
import pytest

@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
def test_add_layer(dtype):
    a = np.random.rand(127, 127).astype(dtype)
    b = np.random.rand(127, 127).astype(dtype)
    out = ginfer_test.run_add_layer_cuda_test(a, b)
    rtol = 1e-2 if dtype == np.float16 else 1e-5
    atol = 1e-1 if dtype == np.float16 else 1e-5
    np.testing.assert_allclose(out, a + b, rtol=rtol, atol=atol)
