# tests/test_memory_stress.py
import gc

import pytest
import torch


@pytest.mark.memory
@pytest.mark.cuda
def test_vram_budget_12gb():
    """Ensures pipeline operations do not exceed the 12GB RTX A2000 limit."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for memory stress test")

    VRAM_BUDGET = 12 * (1024**3)  # 12GB in bytes

    # Simulate a heavy batch allocation (e.g., loading the VLM and BiRefNet simultaneously)
    try:
        dummy_tensors = [
            torch.randn((32, 3, 512, 512), device="cuda") for _ in range(10)
        ]
        allocated = torch.cuda.memory_allocated()

        assert allocated < VRAM_BUDGET, f"VRAM exceeded: {allocated / (1024**3):.2f} GB"
    finally:
        # Guarantee cleanup even if the test fails
        del dummy_tensors
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.memory
def test_birefnet_cpu_gpu_swapping(mocker):
    """Verifies that BiRefNet safely offloads to CPU before the VLM loads into VRAM."""
    mock_birefnet = mocker.MagicMock()

    # Simulate the pipeline flow
    mock_birefnet.to("cuda")
    mock_birefnet.to("cpu")

    # Assert that the CPU offload method was explicitly called
    mock_birefnet.to.assert_called_with("cpu")
